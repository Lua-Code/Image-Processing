import os, cv2
import numpy as np

# ************** PHASE 1: Preprocessing ************** #

def adjust_gamma(image, gamma=1.6):
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255
    table = table.astype("uint8")
    return cv2.LUT(image, table)

def process_puzzle_image(img):
    img_corr = adjust_gamma(img, gamma=1.6)
    gray = cv2.cvtColor(img_corr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, d=9, sigmaColor=40, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4)).apply(blur)
    v = np.median(clahe)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(clahe, lower, upper)
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    adaptive = cv2.adaptiveThreshold(clahe,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,3)
    final = cv2.bitwise_or(closed, adaptive)
    contours, _ = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "original": img,
        "gamma_corrected": img_corr,
        "gray": gray,
        "blur": blur,
        "clahe": clahe,
        "edges": edges,
        "closed": closed,
        "adaptive": adaptive,
        "final": final,
        "contours": contours
    }

# ************** PHASE 2: Segmentation ************** #

def segment_image_maps(maps, split):
    piece_dim = maps["original"].shape[0] // split
    imagePieces = []
    allPieces = {key: [] for key in maps.keys()}
    allPieceBorders = {key: [] for key in maps.keys() if key != "original"}

    index = 0
    for r in range(split):
        for c in range(split):
            y_start = r * piece_dim
            x_start = c * piece_dim

            for key, img in maps.items():
                # use maps[key] as before
                piece = img[y_start:y_start+piece_dim, x_start:x_start+piece_dim]
                allPieces[key].append((index, piece))

                if key != "original":
                    borders = {
                        "top": piece[0, :],
                        "bottom": piece[-1, :],
                        "left": piece[:, 0],
                        "right": piece[:, -1]
                    }
                    allPieceBorders[key].append((index, borders))

            imagePieces.append((index, maps["original"][y_start:y_start+piece_dim, x_start:x_start+piece_dim]))
            index += 1

    return imagePieces, allPieces, allPieceBorders


# ************** PHASE 3: Matching ************** #

def compareBordersAllMaps(pieceBorders1, pieceBorders2, artifactWeights=None):
    if artifactWeights is None:
        artifactWeights = {"edge":1.5,"adaptive":2,"clahe":2,"gammaCorrected":0.8,"bilateral":0.5,"closed":0}

    totalScore = 0
    totalWeight = 0
    for artifact, weight in artifactWeights.items():
        b1 = pieceBorders1[artifact].astype(np.float32)
        b2 = pieceBorders2[artifact].astype(np.float32)
        std1, std2 = np.std(b1), np.std(b2)
        if std1 == 0 and std2 == 0:
            score = 0
        elif std1 == 0 or std2 == 0:
            score = 1e6
        else:
            b1_norm = (b1 - np.mean(b1)) / std1
            b2_norm = (b2 - np.mean(b2)) / std2
            l1_diff = np.mean(np.abs(b1_norm - b2_norm))
            corr = np.corrcoef(b1_norm, b2_norm)[0,1]
            corr_diff = 1 - corr
            score = 0.5*l1_diff + 0.5*corr_diff
        totalScore += score * weight
        totalWeight += weight
    return totalScore / totalWeight

def matchPiecesAllArtifacts(allPieceBorders, artifactWeights=None):
    if artifactWeights is None:
        artifactWeights = {"edge":1.0,"adaptive":1.0,"clahe":1.0,"gammaCorrected":0.8,"bilateral":2.5,"closed":0}
    matches = {}
    keys = list(allPieceBorders.keys())
    n_pieces = len(allPieceBorders[keys[0]])
    combinedBorders = [(allPieceBorders[keys[0]][i][0], {key: allPieceBorders[key][i][1] for key in keys}) for i in range(n_pieces)]
    pieceEdges = {edge:{} for edge in ["top","bottom","left","right"]}
    for i, combined in combinedBorders:
        for edge in ["top","bottom","left","right"]:
            pieceEdges[edge][i] = {artifact: combined[artifact][edge] for artifact in combined}
    for i, _ in combinedBorders:
        best_for_i = {}
        for edge_i_name in ["top","bottom","left","right"]:
            best_score = float('inf')
            best_piece = None
            opposite_edge = {"top":"bottom","bottom":"top","left":"right","right":"left"}[edge_i_name]
            pieceBorders1 = pieceEdges[edge_i_name][i]
            for j, _ in combinedBorders:
                if i==j: continue
                pieceBorders2 = pieceEdges[opposite_edge][j]
                score = compareBordersAllMaps(pieceBorders1,pieceBorders2,artifactWeights)
                if score < best_score:
                    best_score = score
                    best_piece = j
            best_for_i[edge_i_name] = (best_piece, best_score)
        matches[i] = best_for_i
    return matches

# ************** PHASE 4: Grid Assembly ************** #

def assembleBestGrid(matches, split, allPieceBorders):
    best_grid = None
    best_total = float('inf')
    piece_ids = list(matches.keys())
    def edge_cost(a_piece, a_edge, b_piece, b_edge):
        pb1 = {art: allPieceBorders[art][a_piece][1][a_edge] for art in allPieceBorders}
        pb2 = {art: allPieceBorders[art][b_piece][1][b_edge] for art in allPieceBorders}
        return compareBordersAllMaps(pb1,pb2)
    for start in piece_ids:
        grid = [[None for _ in range(split)] for _ in range(split)]
        used = {start}
        grid[0][0] = start
        for r in range(split):
            for c in range(split):
                if r==0 and c==0: continue
                if r==0:
                    left = grid[r][c-1]
                    cand = matches[left]["right"][0]
                    if cand in used or cand is None:
                        for p in piece_ids:
                            if p not in used:
                                cand = p
                                break
                    grid[r][c] = cand
                    used.add(cand)
                    continue
                if c==0:
                    top = grid[r-1][c]
                    cand = matches[top]["bottom"][0]
                    if cand in used or cand is None:
                        for p in piece_ids:
                            if p not in used:
                                cand = p
                                break
                    grid[r][c] = cand
                    used.add(cand)
                    continue
                left = grid[r][c-1]
                top = grid[r-1][c]
                best_piece = None
                best_score = float('inf')
                for p in piece_ids:
                    if p in used: continue
                    s = edge_cost(left,"right",p,"left") + edge_cost(top,"bottom",p,"top")
                    if s < best_score:
                        best_score = s
                        best_piece = p
                grid[r][c] = best_piece
                used.add(best_piece)
        total = 0.0
        for r in range(split):
            for c in range(split):
                if c+1<split: total+=edge_cost(grid[r][c],"right",grid[r][c+1],"left")
                if r+1<split: total+=edge_cost(grid[r][c],"bottom",grid[r+1][c],"top")
        if total < best_total:
            best_total = total
            best_grid = grid
    return best_grid

# ************** PHASE 5: Build Final Image ************** #

def buildFinalImage(imagePieces, indexGrid):
    imageLookup = {i:img for i,img in imagePieces}
    split = len(indexGrid)
    first_img = next(iter(imageLookup.values()))
    piece_h,piece_w = first_img.shape[:2]
    channels = 1 if len(first_img.shape)==2 else first_img.shape[2]
    assembled = np.zeros((piece_h*split, piece_w*split, channels), dtype=np.uint8) if channels>1 else np.zeros((piece_h*split, piece_w*split),dtype=np.uint8)
    for r in range(split):
        for c in range(split):
            piece_index = indexGrid[r][c]
            piece_img = imageLookup[piece_index]
            y,x = r*piece_h, c*piece_w
            assembled[y:y+piece_h,x:x+piece_w] = piece_img
    return assembled

# ************** PHASE 6: Pipeline ************** #

def pipeline(images, puzzle_size="2x2"):
    results = []
    for img_name,img in images:
        maps = process_puzzle_image(img)
        split = int(puzzle_size[0])
        imagePieces, allPieces, allPiecesBorders = segment_image_maps(maps, split)
        matches = matchPiecesAllArtifacts(allPiecesBorders)
        indexGrid = assembleBestGrid(matches, split, allPiecesBorders)
        finalImage = buildFinalImage(imagePieces, indexGrid)
        results.append((img_name, finalImage, maps))
    return results

# ************** TESTING ************** #

def load_images(folder):
    return [(f, cv2.imread(os.path.join(folder,f))) for f in os.listdir(folder) if f.endswith(".jpg")]

# Example run
images = load_images("Gravity Falls/2x2")
pipeline_results = pipeline(images, "2x2")
for name, final, maps in pipeline_results:
    cv2.imshow(name, final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
