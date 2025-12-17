import os
import cv2
import numpy as np
import math
import itertools
from skimage.metrics import structural_similarity as ssim

# =========================
# STEP 1: LOAD / PREPROCESS
# =========================

def adjust_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    return clahe.apply(gray)

def process_puzzle_image(img, split):
    gray = adjust_contrast(img)
    if split > 2:
        smooth = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
    else:
        smooth = cv2.bilateralFilter(gray, d=4, sigmaColor=25, sigmaSpace=25)
    norm = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX)

    if split <= 2:
        sobelx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = cv2.magnitude(sobelx, sobely)
        edge_mag = cv2.normalize(edge_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        v = np.median(norm)
        lower = int(max(0, 0.66*v))
        upper = int(min(255, 1.33*v))
        edge_mag = cv2.Canny(norm, lower, upper, apertureSize=3)

    return {"gray": gray, "smooth": smooth, "norm": norm, "edge": edge_mag}

def getImageMaps(img, split):
    results = process_puzzle_image(img, split)
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
                        "top": piece[0, :].copy(),
                        "bottom": piece[-1, :].copy(),
                        "left": piece[:, 0].copy(),
                        "right": piece[:, -1].copy(),
                    }
                    allPieceBorders[key].append((index, borders))
            imagePieces.append((index, maps["original"][y_start:y_start + pieceDimension, x_start:x_start + pieceDimension]))
            index += 1

    return imagePieces, allPieces, allPieceBorders

# =============================
# STEP 2: PIECE EDGE COMPARISON
# =============================

def compareBordersAllMaps(pieceBorders1, pieceBorders2, patch1=None, patch2=None):
    scores = {}
    eps = 1e-6
    for artifact in pieceBorders1:
        b1 = pieceBorders1[artifact].astype(np.float32).ravel()
        b2 = pieceBorders2[artifact].astype(np.float32).ravel()
        if b1.shape != b2.shape:
            n = min(b1.size, b2.size)
            b1, b2 = b1[:n], b2[:n]

        b1n = (b1 - np.mean(b1)) / (np.std(b1)+eps)
        b2n = (b2 - np.mean(b2)) / (np.std(b2)+eps)
        ssd = np.mean((b1n-b2n)**2)
        corr = np.sum(b1n*b2n)/(b1n.size-1+eps)
        corr_term = 1.0 - corr
        grad_score = np.mean(np.abs(np.diff(b1)/(np.mean(np.abs(np.diff(b1)))+eps) - np.diff(b2)/(np.mean(np.abs(np.diff(b2)))+eps))) if b1.size>1 else 0.0

        hist1 = cv2.calcHist([b1.astype(np.uint8)], [0], None, [16], [0,256])
        hist2 = cv2.calcHist([b2.astype(np.uint8)], [0], None, [16], [0,256])
        hist_score = 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        ssim_score = 0.0
        if patch1 is not None and patch2 is not None:
            try:
                ssim_score = 1.0 - ssim(patch1, patch2, multichannel=True)
            except:
                ssim_score = 0.0

        score = 0.4*corr_term + 0.2*ssd + 0.15*grad_score + 0.15*hist_score + 0.1*ssim_score
        if np.std(b1)<1e-2 and np.std(b2)<1e-2: score *= 0.25
        scores[artifact] = float(score)
        scores["ssim"] = float(ssim_score)
    return scores

def matchPiecesAllArtifacts(allPieceBorders, top_k=12, artifactWeights=None):
    if artifactWeights is None:
        artifactWeights = {"edge":2.0,"norm":1.5,"smooth":1.2,"gray":1.0,"ssim":0.1}
    keys = list(allPieceBorders.keys())
    if not keys: return {}, {}
    n = len(allPieceBorders[keys[0]])
    combinedBorders = [(allPieceBorders[keys[0]][idx][0], {k: allPieceBorders[k][idx][1] for k in keys}) for idx in range(n)]
    pieceEdges = {edge:{} for edge in ["top","bottom","left","right"]}
    for idx, maps in combinedBorders:
        for edge in pieceEdges:
            pieceEdges[edge][idx] = {a: maps[a][edge] for a in maps}
    matches_best = {}
    candidates_per_piece = {}
    for i,_ in combinedBorders:
        best_for_i = {}
        candidates_for_i = {}
        for edge in ["top","bottom","left","right"]:
            opp = {"top":"bottom","bottom":"top","left":"right","right":"left"}[edge]
            borders1 = pieceEdges[edge][i]
            scored = []
            for j,_ in combinedBorders:
                if i==j: continue
                borders2 = pieceEdges[opp][j]
                per_artifact = compareBordersAllMaps(borders1,borders2)
                total = sum(per_artifact[a]*artifactWeights.get(a,1.0) for a in per_artifact)
                scored.append((j,float(total)))
            scored.sort(key=lambda x:x[1])
            candidates_for_i[edge] = scored[:top_k]
            best_for_i[edge] = scored[0] if scored else (None,float("inf"))
        matches_best[i] = best_for_i
        candidates_per_piece[i] = candidates_for_i
    return matches_best, candidates_per_piece

# =========================
# STEP 3: ASSEMBLY
# =========================

def assembleBestGrid(matches_best, candidates_per_piece, split, allPieceBorders):
    piece_ids = list(matches_best.keys())
    
    #========================
    #HELPER FUNCTIONS
    #========================
    def edge_cost(a_piece, a_edge, b_piece, b_edge):
        pb1 = {art: allPieceBorders[art][a_piece][1][a_edge] for art in allPieceBorders}
        pb2 = {art: allPieceBorders[art][b_piece][1][b_edge] for art in allPieceBorders}
        return sum(compareBordersAllMaps(pb1, pb2).values())
    
    
    def copy_grid(grid): return [[cell for cell in row] for row in grid]
    
    
    def get_constraint_score(grid, r, c, piece):
        score,count=0.0,0
        if c>0 and grid[r][c-1] is not None: score+=edge_cost(grid[r][c-1],"right",piece,"left"); count+=1
        if r>0 and grid[r-1][c] is not None: score+=edge_cost(grid[r-1][c],"bottom",piece,"top"); count+=1
        return score/count if count>0 else 0.0
    
    
    def get_candidates(grid,r,c,used,top_k=12):
        candidates=[]
        candidate_set=set()
        neighbor_suggestions=[]
        for dr,dc,edge_from,edge_to in [(-1,0,"bottom","top"),(0,-1,"right","left")]:
            nr,nc=r+dr,c+dc
            if 0<=nr<split and 0<=nc<split and grid[nr][nc] is not None:
                neighbor_piece=grid[nr][nc]
                neighbor_suggestions.extend([p for p,_ in candidates_per_piece[neighbor_piece][edge_from]])
        freq={p:neighbor_suggestions.count(p) for p in neighbor_suggestions if p not in used}
        mutuals=[p for p,f in freq.items() if f>1]
        for p in mutuals: candidate_set.add(p)
        for p in neighbor_suggestions:
            if p in used or p in candidate_set: continue
            candidate_set.add(p)
            if len(candidate_set)>=top_k: break
        for p in list(candidate_set):
            candidates.append((get_constraint_score(grid,r,c,p),p))
        if len(candidates)<top_k:
            remaining=[]
            for p in piece_ids:
                if p in used or p in candidate_set: continue
                ssum=sum(matches_best[p][e][1] for e in ["top","bottom","left","right"])
                remaining.append((ssum,p))
            remaining.sort(key=lambda x:x[0])
            for _,p in remaining[:(top_k-len(candidates))]: candidates.append((get_constraint_score(grid,r,c,p),p))
        candidates.sort(key=lambda x:x[0])
        return [p for _,p in candidates[:top_k]]


    #========================
    #MAIN ASSEMBLY LOGIC
    #=======================
    if split==2:
        best_score=float('inf')
        best_grid=None
        for perm in itertools.permutations(piece_ids):
            grid=[[perm[0],perm[1]],[perm[2],perm[3]]]
            score=0
            for r in range(split):
                for c in range(split):
                    piece=grid[r][c]
                    score+=get_constraint_score(grid,r,c,piece)
            if score<best_score: best_score=score; best_grid=grid
        return best_grid

    beam=[]
    beam_width=300 if split==4 else 600
    for start_piece in piece_ids[:min(len(piece_ids), beam_width*2)]:
        grid=[[None for _ in range(split)] for _ in range(split)]
        grid[0][0]=start_piece
        used={start_piece}
        beam.append((0.0,grid,used))

    best_solution=None
    best_score=float('inf')
    max_steps=split*split-1
    for step in range(max_steps):
        new_beam=[]
        for current_score,grid,used in beam:
            next_r=next_c=None
            for r in range(split):
                for c in range(split):
                    if grid[r][c] is None: next_r,next_c=r,c; break
                if next_r is not None: break
            if next_r is None:
                if current_score<best_score: best_score=current_score; best_solution=grid
                continue
            candidates=get_candidates(grid,next_r,next_c,used,top_k=15 if split<=4 else 30)
            for piece in candidates:
                new_grid=copy_grid(grid)
                new_used=used.copy()
                new_grid[next_r][next_c]=piece
                new_used.add(piece)
                new_score=current_score+get_constraint_score(new_grid,next_r,next_c,piece)
                if new_score>best_score*2.0: continue
                new_beam.append((new_score,new_grid,new_used))
        new_beam.sort(key=lambda x:x[0])
        beam=new_beam[:beam_width]
        if not beam: break

    if best_solution is not None: return best_solution
    if beam: return beam[0][1]
    return [[r*split+c for c in range(split)] for r in range(split)]

# =========================
# FINAL IMAGE BUILDING
# =========================

def buildFinalImage(imagePieces,indexGrid):
    imageLookup={i:img for i,img in imagePieces}
    split=len(indexGrid)
    first_img=next(iter(imageLookup.values()))
    if len(first_img.shape)==2: piece_h,piece_w=first_img.shape; channels=1
    else: piece_h,piece_w,channels=first_img.shape
    if channels==1: assembled=np.zeros((piece_h*split,piece_w*split),dtype=np.uint8)
    else: assembled=np.zeros((piece_h*split,piece_w*split,channels),dtype=np.uint8)
    for r in range(split):
        for c in range(split):
            piece_index=indexGrid[r][c]
            piece_img=imageLookup[piece_index]
            y=r*piece_h; x=c*piece_w
            assembled[y:y+piece_h,x:x+piece_w]=piece_img
    return assembled

# =========================
# PIPELINE FUNCTION
# =========================

def solvePuzzle(img,split,verbose=False):
    maps=getImageMaps(img, split)
    imagePieces,allPiecesBorders,allPieceBorders=segmentImageAndMaps(maps,split)

    if split<=2:
        artifactWeights={"edge":5.0,"norm":1.5,"smooth":1.2,"gray":1.0, "ssim":0.1}
    elif split<=4:
        artifactWeights={"edge":15.0,"norm":8.0,"smooth":4.0,"gray":1.0, "ssim":5.0}
    else:
        artifactWeights={"edge":20.0,"norm":12.0,"smooth":6.0,"gray":2.0, "ssim":8.0}

    matches_best,candidates_per_piece=matchPiecesAllArtifacts(allPieceBorders,top_k=15 if split<=4 else 30,artifactWeights=artifactWeights)
    indexGrid=assembleBestGrid(matches_best,candidates_per_piece,split,allPieceBorders)
    finalImage=buildFinalImage(imagePieces,indexGrid)
    return finalImage

# =========================
# TESTING FUNCTIONS
# =========================

#test with visual output
def manualTesting(dataPath="Gravity Falls/4x4", total_images=110):
    if "2x2" in dataPath: split=2
    elif "4x4" in dataPath: split=4
    else: split=8

    for i in range(total_images):
        imgPath=os.path.join(dataPath,f"{i}.jpg")
        img=cv2.imread(imgPath)
        if img is None: continue
        result=solvePuzzle(img,split)
        cv2.imshow(f"Reconstructed {i}", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def solveOneImage(imgPath, split=4):
    img = cv2.imread(imgPath)
    if img is None:
        print(f"Image not found: {imgPath}")
        return
    
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)


    maps = getImageMaps(img, split)
    for map_name, map_img in maps.items():
        if(map_name=="original"): continue
        title = f"Map: {map_name}"


        display_img = map_img

        cv2.imshow(title, display_img)
        cv2.waitKey(0)

    finalImage = solvePuzzle(img, split)

    cv2.imshow("Reconstructed Puzzle", finalImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Finished processing {imgPath}")



#returns an accuracy value
def scoreAllImages(dataPath="Gravity Falls/4x4",correctPath="correct",total_images=110, threshold=0.93):
    if "2x2" in dataPath: split=2
    elif "4x4" in dataPath: split=4
    else: split=8
    correct_count=0
    for i in range(total_images):
        imgPath=os.path.join(dataPath,f"{i}.jpg")
        correctImgPath=os.path.join(correctPath,f"{i}.png")
        img=cv2.imread(imgPath)
        correct_img=cv2.imread(correctImgPath)
        if img is None or correct_img is None: continue
        reconstructed=solvePuzzle(img,split)
        if reconstructed.shape!=correct_img.shape:
            reconstructed=cv2.resize(reconstructed,(correct_img.shape[1],correct_img.shape[0]))
        final_gray=cv2.cvtColor(reconstructed,cv2.COLOR_BGR2GRAY) if len(reconstructed.shape)==3 else reconstructed
        correct_gray=cv2.cvtColor(correct_img,cv2.COLOR_BGR2GRAY) if len(correct_img.shape)==3 else correct_img
        score=ssim(final_gray,correct_gray)
        if score>=threshold: correct_count+=1
    print(f"Correctly reconstructed images: {correct_count}/{total_images}")
    print(f"Accuracy: {math.floor(correct_count/total_images*100)}%")
   
    
def visualizeMatches(img, split=4, top_k=5):
    maps = getImageMaps(img, split)
    imagePieces, allPiecesBorders, allPieceBorders = segmentImageAndMaps(maps, split)
    artifactWeights = {"edge":15.0,"norm":8.0,"smooth":4.0,"gray":1.0, "ssim":5.0} if split>2 else {"edge":5.0,"norm":1.5,"smooth":1.2,"gray":1.0,"ssim":0.1}

    matches_best, candidates_per_piece = matchPiecesAllArtifacts(allPieceBorders, top_k=top_k, artifactWeights=artifactWeights)
    pieceDimension = img.shape[0] // split
    canvas = img.copy()

    centers = {}
    for idx, _ in enumerate(imagePieces):
        r = idx // split
        c = idx % split
        y = r*pieceDimension + pieceDimension//2
        x = c*pieceDimension + pieceDimension//2
        centers[idx] = (x, y)

    for i in candidates_per_piece:
        print(f"Piece {i} matches:")
        for edge, candidates in candidates_per_piece[i].items():
            opp_edge = {"top":"bottom","bottom":"top","left":"right","right":"left"}[edge]
            for j, score in candidates[:top_k]:
                if i >= j:  
                    continue
                pt1 = centers[i]
                pt2 = centers[j]
                cv2.arrowedLine(canvas, pt1, pt2, (0, 255, 0), 2, tipLength=0.2)
                print(f"  Edge '{edge}' -> Piece {j}, Score: {score:.4f}")

    cv2.imshow(f"Candidate Matches (top {top_k})", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================

if __name__=="__main__":
    #   Example: manual testing
    #manualTesting("Gravity Falls/2x2")
    #manualTesting("Gravity Falls/4x4")
    #manualTesting("Gravity Falls/8x8")
    
    #   Example: scoring
    scoreAllImages("Gravity Falls/2x2","correct")
    #scoreAllImages("Gravity Falls/4x4","correct")
    #scoreAllImages("Gravity Falls/8x8","correct")
    
    #   Example: visualize matches
    #visualizeMatches(cv2.imread("Gravity Falls/2x2/0.jpg"), split=2, top_k=2)
    #visualizeMatches(cv2.imread("Gravity Falls/4x4/0.jpg"), split=4, top_k=4)
    #visualizeMatches(cv2.imread("Gravity Falls/8x8/0.jpg"), split=8, top_k=2)
    
    #   Example: solve one image with detailed map visualization
    #solveOneImage("Gravity Falls/2x2/2.jpg", split=2)
    #solveOneImage("Gravity Falls/4x4/1.jpg", split=4)

