import os
import cv2
import numpy as np
import math

# =========================
# PHASE 1: LOAD / PREPROCESS
# =========================

def importImages():
    dataPath = "Gravity Falls"
    folders = ["2x2", "4x4", "8x8"]
    images = {"2x2": [], "4x4": [], "8x8": []}

    for folder in folders:
        folderPath = os.path.join(dataPath, folder)
        for fileName in os.listdir(folderPath):
            imgPath = os.path.join(folderPath, fileName)
            img = cv2.imread(imgPath)
            if img is None:
                continue
            images[folder].append((fileName, img))
    return images

def adjust_gamma(image, gamma=1.5):
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255
    return cv2.LUT(image, table.astype("uint8"))

def process_puzzle_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=20)
    norm = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX)
    sobelx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sobelx, sobely)
    edge_mag = cv2.normalize(edge_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {"gray": gray, "smooth": smooth, "norm": norm, "edge": edge_mag}

# =========================
# PHASE 2: MAPS + SEGMENTATION
# =========================

def getImageMaps(img):
    results = process_puzzle_image(img)
    return {
        "original": img,
        "gray": results["gray"],
        "smooth": results["smooth"],
        "norm": results["norm"],
        "edge": results["edge"],
    }

def segmentImageAndMaps(maps, split):
    pieceDimension = maps["original"].shape[0] // split
    index = 0
    imagePieces = []
    allPieces = {key: [] for key in maps.keys()}
    allPieceBorders = {key: [] for key in maps.keys() if key != "original"}

    for r in range(split):
        for c in range(split):
            y_start = r * pieceDimension
            x_start = c * pieceDimension

            for key, img in maps.items():
                piece = img[y_start:y_start + pieceDimension, x_start:x_start + pieceDimension]
                allPieces[key].append((index, piece))

                if key != "original":
                    borders = {
                        "top": piece[0, :],
                        "bottom": piece[-1, :],
                        "left": piece[:, 0],
                        "right": piece[:, -1],
                    }
                    allPieceBorders[key].append((index, borders))

            imagePieces.append((index, maps["original"][y_start:y_start + pieceDimension, x_start:x_start + pieceDimension]))
            index += 1

    return imagePieces, allPieces, allPieceBorders

# =========================
# PHASE 3: MATCHING (FIXED STABILITY)
# =========================

def compareBordersAllMaps(pieceBorders1, pieceBorders2):
    scores = {}

    for artifact in pieceBorders1:
        b1 = pieceBorders1[artifact].astype(np.float32)
        b2 = pieceBorders2[artifact].astype(np.float32)

        std1, std2 = np.std(b1), np.std(b2)

        if std1 == 0 and std2 == 0:
            scores[artifact] = 0.0
            continue
        if std1 == 0 or std2 == 0:
            scores[artifact] = 1e6
            continue

        b1n = (b1 - np.mean(b1)) / std1
        b2n = (b2 - np.mean(b2)) / std2

        l1 = np.mean(np.abs(b1n - b2n))

        try:
            corr = np.corrcoef(b1n, b2n)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0

        corr_diff = 1.0 - corr
        scores[artifact] = 0.5 * l1 + 0.5 * corr_diff

    return scores

def matchPiecesAllArtifacts(allPieceBorders, artifactWeights=None):
    if artifactWeights is None:
        artifactWeights = {"edge": 2.0, "norm": 1.5, "smooth": 1.2, "gray": 1.0}

    matches = {}
    keys = list(allPieceBorders.keys())
    n = len(allPieceBorders[keys[0]])

    combinedBorders = [
        (allPieceBorders[keys[0]][idx][0], {k: allPieceBorders[k][idx][1] for k in keys})
        for idx in range(n)
    ]

    pieceEdges = {edge: {} for edge in ["top", "bottom", "left", "right"]}
    for idx, maps in combinedBorders:
        for edge in pieceEdges:
            pieceEdges[edge][idx] = {a: maps[a][edge] for a in maps}

    for i, _ in combinedBorders:
        best_for_i = {}
        for edge in ["top", "bottom", "left", "right"]:
            opp = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[edge]
            borders1 = pieceEdges[edge][i]

            best_score = float("inf")
            best_piece = None

            for j, _ in combinedBorders:
                if i == j:
                    continue

                borders2 = pieceEdges[opp][j]
                per_artifact = compareBordersAllMaps(borders1, borders2)
                total = sum(per_artifact[a] * artifactWeights.get(a, 1.0) for a in per_artifact)

                if total < best_score:
                    best_score = total
                    best_piece = j

            best_for_i[edge] = (best_piece, best_score)

        matches[i] = best_for_i

    return matches

# =========================
# PHASE 4: ASSEMBLY + RECONSTRUCTION
# =========================

def assembleBestGrid(matches, split, allPieceBorders, beam_width=7):
    """
    Beam search assembly: maintains top-k partial solutions at each step.
    Uses only standard Python - no external imports needed.
    """
    piece_ids = list(matches.keys())
    
    def edge_cost(a_piece, a_edge, b_piece, b_edge):
        pb1 = {art: allPieceBorders[art][a_piece][1][a_edge] for art in allPieceBorders}
        pb2 = {art: allPieceBorders[art][b_piece][1][b_edge] for art in allPieceBorders}
        return sum(compareBordersAllMaps(pb1, pb2).values())
    
    def copy_grid(grid):
        """Deep copy a 2D grid."""
        return [[cell for cell in row] for row in grid]
    
    def get_constraint_score(grid, r, c, piece):
        """Calculate how well a piece fits based on already-placed neighbors."""
        score = 0.0
        count = 0
        
        # Check left neighbor
        if c > 0 and grid[r][c-1] is not None:
            score += edge_cost(grid[r][c-1], "right", piece, "left")
            count += 1
        
        # Check top neighbor
        if r > 0 and grid[r-1][c] is not None:
            score += edge_cost(grid[r-1][c], "bottom", piece, "top")
            count += 1
        
        # Check right neighbor (if already placed)
        if c < split - 1 and grid[r][c+1] is not None:
            score += edge_cost(piece, "right", grid[r][c+1], "left")
            count += 1
        
        # Check bottom neighbor (if already placed)
        if r < split - 1 and grid[r+1][c] is not None:
            score += edge_cost(piece, "bottom", grid[r+1][c], "top")
            count += 1
        
        return score / count if count > 0 else 0.0
    
    def get_candidates(grid, r, c, used, top_k=3):

        candidates = []
        
        suggestions = set()
        if c > 0 and grid[r][c-1] is not None:
            left_piece = grid[r][c-1]
            suggestions.add(matches[left_piece]["right"][0])
        if r > 0 and grid[r-1][c] is not None:
            top_piece = grid[r-1][c]
            suggestions.add(matches[top_piece]["bottom"][0])
        
        for piece in suggestions:
            if piece is not None and piece not in used:
                score = get_constraint_score(grid, r, c, piece)
                candidates.append((score, piece))
        
        for piece in piece_ids:
            if piece not in used and piece not in suggestions:
                score = get_constraint_score(grid, r, c, piece)
                candidates.append((score, piece))
        
        candidates.sort(key=lambda x: x[0])
        return [p for _, p in candidates[:top_k]]
    
    beam = []
    for start_piece in piece_ids[:min(beam_width, len(piece_ids))]:
        grid = [[None for _ in range(split)] for _ in range(split)]
        grid[0][0] = start_piece
        used = {start_piece}
        beam.append((0.0, grid, used))  
    
    best_solution = None
    best_score = float('inf')
    
    for step in range(split * split - 1): 
        new_beam = []
        
        for current_score, grid, used in beam:
            next_r, next_c = None, None
            for r in range(split):
                for c in range(split):
                    if grid[r][c] is None:
                        next_r, next_c = r, c
                        break
                if next_r is not None:
                    break
            
            if next_r is None: 
                if current_score < best_score:
                    best_score = current_score
                    best_solution = grid
                continue
            
            candidates = get_candidates(grid, next_r, next_c, used, top_k=3)
            
            for piece in candidates:
                new_grid = copy_grid(grid)
                new_used = used.copy()
                new_grid[next_r][next_c] = piece
                new_used.add(piece)
                
                piece_score = get_constraint_score(new_grid, next_r, next_c, piece)
                new_score = current_score + piece_score
                
                new_beam.append((new_score, new_grid, new_used))
        
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
        if not beam:
            break
    
    if best_solution:
        return best_solution
    
    if beam:
        return beam[0][1]




def buildFinalImage(imagePieces, indexGrid):
    imageLookup = {i: img for i, img in imagePieces}
    split = len(indexGrid)
    first_img = next(iter(imageLookup.values()))

    if len(first_img.shape) == 2:
        piece_h, piece_w = first_img.shape
        channels = 1
    else:
        piece_h, piece_w, channels = first_img.shape

    if channels == 1:
        assembled = np.zeros((piece_h * split, piece_w * split), dtype=np.uint8)
    else:
        assembled = np.zeros((piece_h * split, piece_w * split, channels), dtype=np.uint8)

    for r in range(split):
        for c in range(split):
            piece_index = indexGrid[r][c]
            piece_img = imageLookup[piece_index]
            y = r * piece_h
            x = c * piece_w
            assembled[y:y + piece_h, x:x + piece_w] = piece_img

    return assembled

def isCorrect(final_image, correct_image, threshold=70):
    if final_image.shape != correct_image.shape:
        final_image = cv2.resize(final_image, (correct_image.shape[1], correct_image.shape[0]))
    mse = np.mean((final_image.astype(np.float32) - correct_image.astype(np.float32)) ** 2)
    return mse < threshold

# =========================
# PIPELINE + TESTS (UNCHANGED STRUCTURE)
# =========================

def solvePuzzle(img, split, correct_image=None, threshold=200, verbose=False):
    maps = getImageMaps(img)
    imagePiece, allPiecesBorders, allPieceBorders = segmentImageAndMaps(maps, split)
    if verbose:
        print(f"\nWe have {len(imagePiece)} pieces (should be {split*split})")
        for idx, _ in imagePiece:
            print(f"Piece {idx}")

    matches = matchPiecesAllArtifacts(allPieceBorders)
    if verbose:
        print("\n=== DETAILED MATCH ANALYSIS ===")
        for i in range(split * split):
            edges = matches[i]
            print(f"\n🧩 Piece {i}:")
            print(f" TOP → Piece {edges['top'][0]:2d} | Score: {edges['top'][1]:8.2f}")
            print(f" BOTTOM → Piece {edges['bottom'][0]:2d} | Score: {edges['bottom'][1]:8.2f}")
            print(f" LEFT → Piece {edges['left'][0]:2d} | Score: {edges['left'][1]:8.2f}")
            print(f" RIGHT → Piece {edges['right'][0]:2d} | Score: {edges['right'][1]:8.2f}")

    indexGrid = assembleBestGrid(matches, split, allPieceBorders)
    if verbose:
        print("\n=== Final Grid ===")
        for row in indexGrid:
            print(row)

    finalImage = buildFinalImage(imagePiece, indexGrid)

    result = {"reconstructed": finalImage, "grid": indexGrid, "maps": maps}
    if correct_image is not None:
        result["is_correct"] = isCorrect(finalImage, correct_image, threshold)
    return result

def manualTesting():
    dataPath = "Gravity Falls/4x4"
    correctPath = "correct"
    size = 2
    
    if "2x2" in dataPath:
        size = 2
    elif "4x4" in dataPath:
        size = 4
    else:
        size = 8

    for imageName in range(110):
        imgPath = os.path.join(dataPath, f"{imageName}.jpg")
        correctImgPath = os.path.join(correctPath, f"{imageName}.png")

        img = cv2.imread(imgPath)
        correct_img = cv2.imread(correctImgPath)
        if img is None or correct_img is None:
            continue

        result = solvePuzzle(img, size, correct_image=correct_img, verbose=True)

        cv2.imshow(str(imageName), result["reconstructed"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if result["is_correct"]:
            print(f"✓ Image {imageName} correct")
        else:
            print(f"✗ Image {imageName} incorrect")

def scoreAllImages():
    dataPath = "Gravity Falls/2x2"
    correctPath = "correct"
    total_images = 110
    correct_count = 0
    wrongImages = []

    if "2x2" in dataPath:
        size = 2
    elif "4x4" in dataPath:
        size = 4
    else:
        size = 8

    for i in range(total_images):
        puzzle_name = str(i)
        imgPath = os.path.join(dataPath, f"{puzzle_name}.jpg")
        correctImgPath = os.path.join(correctPath, f"{puzzle_name}.png")

        img = cv2.imread(imgPath)
        correct_img = cv2.imread(correctImgPath)
        if img is None or correct_img is None:
            continue

        result = solvePuzzle(img, split=size, correct_image=correct_img, threshold=200)
        if result["is_correct"]:
            correct_count += 1
        else:
            wrongImages.append(puzzle_name)

    print(f"Correctly reconstructed images: {correct_count}/{total_images}")
    print(f"Accuracy is: {math.floor(correct_count / total_images * 100)}%")
    for imageName in wrongImages:
        print(imageName)

# TESTING 
scoreAllImages()
#manualTesting()
