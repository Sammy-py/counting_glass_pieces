import cv2
import numpy as np
import matplotlib.pyplot as plt


#########IMAGE-PROSESSING################

img = cv2.imread('images/glass5.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (11,11), 0)

canny = cv2.Canny(blur, 30, 150, 3)

dilated = cv2.dilate(canny, (1,1), iterations = 2)



#########IMAGE-BORDER-MERGE################

(cnt,_)=cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img,cnt,-1,(0,255,0),2)


print('glass count: ', len(cnt))


cv2.imwrite("result/glass5-result.jpg" , img)


cv2.destroyAllWindows()



