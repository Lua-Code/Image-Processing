import os,cv2
import numpy as np

#**************PHASE 1****************# 

#Import the Image Files first :3
def importImages():
    dataPath = "Gravity Falls"
    folders = ["2x2","4x4","8x8"]

    images = {
        "2x2": [],
        "4x4": [],
        "8x8": []
    }

    for folder in folders:
        folderPath = os.path.join(dataPath,folder)

        for fileName in os.listdir(folderPath):
            imgPath = os.path.join(folderPath,fileName)
            img = cv2.imread(imgPath)
            images[folder].append((fileName,img))
            
    return images
    #for folder_name, imgs in images.items():
    #    print(f"{folder_name}: {len(imgs)} images loaded")


def adjust_gamma(image, gamma=1.5):
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255
    table = table.astype("uint8")
    return cv2.LUT(image, table)


def process_puzzle_image(img):

    # Apply Gamma Correction
    img_corr = adjust_gamma(img, gamma=1.6)

    # Convert to Grayscale
    gray = cv2.cvtColor(img_corr, cv2.COLOR_BGR2GRAY)

    # Apply Bilateral Filter
    blur = cv2.bilateralFilter(gray, d=9, sigmaColor=40, sigmaSpace=75)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4)).apply(blur)

    # Apply Auto Canny
    v = np.median(clahe)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(clahe, lower, upper)

    # Apply Morphological Closing
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(clahe,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,3)

    # Combine Closed and Adaptive results
    final = cv2.bitwise_or(closed, adaptive)
    contours, _ = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "gamma_corrected":img_corr,
        "gray": gray,
        "clahe": clahe,
        "blur" : blur,
        "edges": edges,
        "closed": closed,
        "adaptive": adaptive,
        "final": final,
        "contours": contours
    }
    
def save_pipeline_results(imgName, img, results,puzzleSize):
    
    img_base = os.path.splitext(imgName)[0]  
    folder_path = os.path.join("Outputs",puzzleSize, img_base)
    os.makedirs(folder_path, exist_ok=True)
    
    #Save results to file 'Output'
    cv2.imwrite(os.path.join(folder_path, "original.png"), img)
    cv2.imwrite(os.path.join(folder_path, "gamma_corrected.png"), results["gamma_corrected"])
    cv2.imwrite(os.path.join(folder_path, "gray.png"), results["gray"])
    cv2.imwrite(os.path.join(folder_path, "bilateral.png"), results["blur"])
    cv2.imwrite(os.path.join(folder_path, "clahe.png"), results["clahe"])
    cv2.imwrite(os.path.join(folder_path, "edges.png"), results["edges"])
    cv2.imwrite(os.path.join(folder_path, "closed.png"), results["closed"])    
    cv2.imwrite(os.path.join(folder_path, "adaptive.png"), results["adaptive"])
    cv2.imwrite(os.path.join(folder_path, "final.png"), results["final"])    
    
    contourImg = img.copy()
    cv2.drawContours(contourImg, results["contours"], -1, (0,255,0), 2)
    cv2.imwrite(os.path.join(folder_path, "contours.png"), contourImg)
    
    print(f"Saved pipeline results for {imgName} in {folder_path} meow meow")    
    

def show_pipeline_results(img, results):
    
    # Display results
    cv2.imshow("Original", img)
    cv2.imshow("Gamma Corrected",results["gamma_corrected"])
    cv2.imshow("Grayscale", results["gray"])
    cv2.imshow("blur", results["blur"])
    cv2.imshow("CLAHE", results["clahe"])
    cv2.imshow("Edges (Canny)", results["edges"])
    cv2.imshow("Closed Morphology",results["closed"])
    cv2.imshow("Adaptive Thresholding", results["adaptive"])
    cv2.imshow("Final Image",results["final"])
    
    contour_img = img.copy()
    cv2.drawContours(contour_img, results["contours"], -1, (0,255,0), 2)
    cv2.imshow("Contours", contour_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
#display all steps for an image tupple -> <filename,image>
def show_image(imageTupple):
    results = process_puzzle_image(imageTupple[1])
    show_pipeline_results(imageTupple[1], results)

#process all images and save them
def process_and_save_images(images):
    for imageSize, imageList in images.items():
        for imageTuple in imageList:
           results = process_puzzle_image(imageTuple[1])
           save_pipeline_results(imageTuple[0], imageTuple[1], results, imageSize)
           
    
#**************PHASE 2****************#  


#Retrieve the Edge Map for the image we want to assemble
def getImageMaps(filename, size):
    processedImages = "Outputs"
    imageSizeFolderPath = os.path.join(processedImages, size)
    img_base = os.path.splitext(filename)[0] 
    processedImageFolder = os.path.join(imageSizeFolderPath, img_base)
    edgePath = os.path.join(processedImageFolder, "edges.png")
    adaptivePath = os.path.join(processedImageFolder, "adaptive.png")
    gamaCorrectedPath = os.path.join(processedImageFolder, "gamma_corrected.png")
    clahePath = os.path.join(processedImageFolder, "clahe.png")
    bilateralPath = os.path.join(processedImageFolder, "bilateral.png")
    imagePath = os.path.join(processedImageFolder, "original.png")
    
    image = cv2.imread(imagePath)
    edgeMap = cv2.imread(edgePath, cv2.IMREAD_GRAYSCALE)
    adaptiveMap = cv2.imread(adaptivePath, cv2.IMREAD_GRAYSCALE)
    gammaCorrectedMap = cv2.imread(gamaCorrectedPath, cv2.IMREAD_GRAYSCALE)
    claheMap = cv2.imread(clahePath, cv2.IMREAD_GRAYSCALE)
    bilateralMap = cv2.imread(bilateralPath, cv2.IMREAD_GRAYSCALE)
    
    maps = {
        "original" : image,
        "edge": edgeMap,
        "adaptive" : adaptiveMap,
        "gammaCorrected" : gammaCorrectedMap,
        "clahe" : claheMap,
        "bilateral" : bilateralMap
    }
    
    if edgeMap is None:
        raise FileNotFoundError(f"Edge map not found at {edgePath}")
    
    if size == "2x2":
        return maps,2
    elif size == "4x4":
        return maps,4
    else:
        return maps,8        
        
    
# Segment Pieces and split them toe
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
                piece = img[y_start:y_start+pieceDimension, x_start:x_start+pieceDimension]
                allPieces[key].append((index, piece))

                if key != "original":
                    borders = {
                        "top": piece[0, :],
                        "bottom": piece[-1, :],
                        "left": piece[:, 0],
                        "right": piece[:, -1]
                    }
                    allPieceBorders[key].append((index, borders))

            imagePieces.append((index, maps["original"][y_start:y_start+pieceDimension, x_start:x_start+pieceDimension]))
            index += 1

    return imagePieces, allPieces, allPieceBorders




def compareBordersAllMaps(pieceBorders1, pieceBorders2, artifactWeights=None):
    if artifactWeights is None:
        artifactWeights = {
            "edge": 0.0,     
            "adaptive": 1.5, 
            "clahe": 0.0, 
            "gammaCorrected": 0.5,
            "bilateral": 0.3
        }

    totalScore = 0
    totalWeight = 0

    for artifact, weight in artifactWeights.items():
        b1 = pieceBorders1[artifact].astype(np.float32)
        b2 = pieceBorders2[artifact].astype(np.float32)[::-1]  


        std1, std2 = np.std(b1), np.std(b2)
        if std1 == 0 and std2 == 0:
            score = 0
        elif std1 == 0 or std2 == 0:
            score = 1e6  
        else:
            b1 = (b1 - np.mean(b1)) / std1
            b2 = (b2 - np.mean(b2)) / std2
            score = np.mean(np.abs(b1 - b2))  

        totalScore += score * weight
        totalWeight += weight

    return totalScore / totalWeight





def matchPiecesAllArtifacts(allPieceBorders, artifactWeights=None):
    matches = {}

    keys = list(allPieceBorders.keys())
    combinedBorders = []
    for idx in range(len(allPieceBorders[keys[0]])):
        index = allPieceBorders[keys[0]][idx][0]
        combined = {}
        for key in keys:
            combined[key] = allPieceBorders[key][idx][1]
        combinedBorders.append((index, combined))

    for i, borders_i in combinedBorders:
        best_for_i = {}

        for edge_i_name in ["top", "bottom", "left", "right"]:
            best_score = float('inf')
            best_piece = None
            opposite_edge = {
                "top": "bottom",
                "bottom": "top",
                "left": "right",
                "right": "left"
            }[edge_i_name]

            for j, borders_j in combinedBorders:
                if i == j:
                    continue

                pieceBorders1 = {artifact: borders_i[artifact][edge_i_name] for artifact in borders_i}
                pieceBorders2 = {artifact: borders_j[artifact][opposite_edge] for artifact in borders_j}

                score = compareBordersAllMaps(pieceBorders1, pieceBorders2, artifactWeights)

                if score < best_score:
                    best_score = score
                    best_piece = j

            best_for_i[edge_i_name] = (best_piece, best_score)

        matches[i] = best_for_i

    for k, v in matches.items():
        print(f"Piece {k}: {v}")

    return matches




def findTopLeftCorner(matches, artifactWeights=None):
    corner_scores = {}

    for i, edges in matches.items():
        score_top = edges["top"][1]
        score_left = edges["left"][1]

        if score_top == float('inf'):
            score_top = 1e6
        if score_left == float('inf'):
            score_left = 1e6

        corner_scores[i] = score_top + score_left

    topLeft = max(corner_scores, key=corner_scores.get)
    return topLeft





def assembleIndexArray(matches, split):
    grid = [[None for _ in range(split)] for _ in range(split)]
    used = set()

    start = findTopLeftCorner(matches)
    grid[0][0] = start
    used.add(start)

    for r in range(split):
        for c in range(split):
            if r == 0 and c == 0:
                continue

            candidates = []

            if c > 0:
                left_piece = grid[r][c - 1]
                candidate, score = matches[left_piece]["right"]
                if candidate not in used and candidate is not None:
                    candidates.append((candidate, score))

            if r > 0:
                top_piece = grid[r - 1][c]
                candidate, score = matches[top_piece]["bottom"]
                if candidate not in used and candidate is not None:
                    candidates.append((candidate, score))

            if candidates:
                chosen = min(candidates, key=lambda x: x[1])[0]
            else:
                chosen = next(i for i in matches.keys() if i not in used)

            grid[r][c] = chosen
            used.add(chosen)

    return grid


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

            assembled[y:y+piece_h, x:x+piece_w] = piece_img

    return assembled





#UTILITY

def showPieces(pieces):
    for piece in pieces:
        cv2.imshow()


    
#TEST
maps, split = getImageMaps("0", "2x2")
imagePiece, allPieces, allPiecesBorders = segmentImageAndMaps(maps, split)

# Debug: Show what pieces we have
print(f"\nWe have {len(imagePiece)} pieces (should be {split*split})")
for idx, _ in imagePiece:
    print(f"Piece {idx}")

matches = matchPiecesAllArtifacts(allPiecesBorders)

# Debug: Analyze match scores
print("\n=== DETAILED MATCH ANALYSIS ===")
for i in range(split * split):
    edges = matches[i]
    print(f"\n🧩 Piece {i}:")
    print(f"   TOP    → Piece {edges['top'][0]:2d} | Score: {edges['top'][1]:8.2f}")
    print(f"   BOTTOM → Piece {edges['bottom'][0]:2d} | Score: {edges['bottom'][1]:8.2f}")
    print(f"   LEFT   → Piece {edges['left'][0]:2d} | Score: {edges['left'][1]:8.2f}")
    print(f"   RIGHT  → Piece {edges['right'][0]:2d} | Score: {edges['right'][1]:8.2f}")
    
    # Calculate corner score
    corner_score = edges['top'][1] + edges['left'][1]
    print(f"   Corner Score (top+left): {corner_score:.2f}")

# Now run corner detection
topLeft = findTopLeftCorner(matches)
print(f"\n🎯 Selected top-left corner: Piece {topLeft}")

indexGrid = assembleIndexArray(matches, split)
print("\n=== Final Grid ===")
for row in indexGrid:
    print(row)

finalImage = buildFinalImage(imagePiece, indexGrid)
cv2.imshow("Reconstructed Puzzle", finalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()










    
















