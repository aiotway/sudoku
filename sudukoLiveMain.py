import sudukoSolverFast
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
import signal

########################################################################
model = intializePredectionModel()  # LOAD THE CNN MODEL
heightImg = 480
widthImg = 640
 # VIDEO
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('sudoku_solver.avi', fourcc, 20.0, (640, 480))
solving = False
delay = 0
timer = 0
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
########################################################################


while True:
    success, frame = cap.read()
    show = frame
    imgThreshold = preProcess(frame)
    ### 2. FIND ALL COUNTOURS
    imgContours = frame.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = frame.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    ### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
    biggest, maxArea = biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    if maxArea > 9000 and biggest.size > 0:
        biggest = reorder(biggest)
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
        imgWarpColored = cv2.warpPerspective(frame, matrix, (widthImg, heightImg))
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        timer = 0
        delay = delayHandle(delay)
        text = "Solve"
        if solving is False and delay > 10:
            #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
            imgSolvedDigits = imgBlank.copy()
            imgWarpColored = cv2.resize(imgWarpColored, (450, 450))
            boxes = splitBoxes(imgWarpColored)
            numbers = getPredectionStr(boxes, model)
            if len(numbers) == 81:
                solving = True
                numbers = np.asarray(numbers)
                posArray = np.where(numbers != '0', 0, 1)
                ### 5. FIND SOLUTION OF THE BOARD
                signal.signal(signal.SIGALRM, handler)
                # Define a timeout for your function
                signal.alarm(3)
                try:
                    board = sudukoSolverFast.solve(numbers)
                except:
                    solving = False
                signal.alarm(0)
                if type(board) is bool:
                    solving = board
        if solving:
            flatList = []
            for k, v in board.items():
                flatList.append(int(v))
            solvedNumbers = flatList * posArray
            imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)
            #### 6. OVERLAY SOLUTION
            pts2 = np.float32(biggest)  # PREPARE POINTS FOR WARP
            pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
            imgInvWarpColored = frame.copy()
            imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
            inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, frame, 1, 1)
            show = inv_perspective
    else:
        text = "No Sudoku Found"
        if solving:
            delay, solving, timer = resetStatus(delay, solving, timer)
    cv2.putText(show, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 0, 0))
    cv2.imshow("Sudoku show", show)
    out.write(show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the window / Release webcam
cap.release()

# After we release our webcam, we also release the output
out.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()