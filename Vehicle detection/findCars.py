import cv2
import time
import numpy as np

DELAY = 0.02
IS_FOUND = 0

MORPH = 7
CANNY = 250

_width  = 1280
_height = 720
_margin = 0.0

corners = np.array(
	[
		[[  		_margin, _margin 			]],
		[[ 			_margin, _height + _margin  ]],
		[[ _width + _margin, _height + _margin  ]],
		[[ _width + _margin, _margin 			]],
	]
)

pts_dst = np.array( corners, np.float32 )
cap = cv2.VideoCapture('high.flv')

while True :
    ret, frame = cap.read()
    
    # the end of video
    if not ret:
        break
    
    # turn to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter( gray, 1, 10, 120 )
    edges  = cv2.Canny( gray, 20, CANNY )
    
    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( MORPH, MORPH ) )
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    fgmask, contours, h = cv2.findContours( closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    for cont in contours:
        # ignore small rect
        area = cv2.contourArea(cont)
        if area > 10 and area < 500:
            arc_len = cv2.arcLength( cont, True )
            approx = cv2.approxPolyDP( cont, 0.1 * arc_len, True )
            if ( len( approx ) = 4 ):
                pts_src = np.array( approx, np.float32 )
                h, status = cv2.findHomography( pts_src, pts_dst )
                out = cv2.warpPerspective( frame, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )
                cv2.drawContours( frame, [approx], -1, ( 255, 0, 0 ), 2 )
            else : pass
    #cv2.imshow( 'closed', closed )
    #cv2.imshow( 'gray', gray )
    #cv2.namedWindow( 'edges', cv2.CV_WINDOW_AUTOSIZE )
    cv2.imshow( 'edges', edges )
    
    #cv2.namedWindow( 'frame', cv2.CV_WINDOW_AUTOSIZE )
    cv2.imshow( 'rgb', frame )

    key = cv2.waitKey(1) & 0xFF
    #Press ESC key
    if key == 27:
        break
 
#cleat and destory
cap.release()
cv2.destroyAllWindows()
