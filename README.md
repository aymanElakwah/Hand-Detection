Introduction
============

Hand detection has been an interesting topic in the last few years.
There are many approaches for hand detection, many of them approved
their excellence especially deep learning based approaches. Traditional
image processing techniques also can get acceptable results, that’s what
we demonstrate here.

We propose a traditional image processing technique for detecting hand
and different gestures. Then, we use it in controlling the mouse.

Proposed Method
===============

Overview
--------

1.  Start with a calibration to specify the range of skin color.

2.  Detect the motion in the current frame relative to the
    previous frames.

3.  Detect the face using Haar Cascade face detector and remove it from
    the original frame.

4.  Segment the frame resulted from the previous step, based on the
    detected skin color range in the first step.

5.  Combine the masks made in the second and fourth step to get the
    final image.

6.  Perform gesture recognition on the result image from step 5.

![image](https://user-images.githubusercontent.com/32196766/71312996-3dfeca80-243b-11ea-8905-d11948ef34e5.png)

Calibration
-----------

It is done by waving with the front hand then the back hand while the
scene is static and nothing is moving except for the hand.

We use the motion detection module which we will talk about in the next
subsection to know the moving object here (the hand only in this case).

Then convert the frame from RGB color space to YCrCb color frame and
construct a histogram for the area of motion.

We take the mode of this histogram (the most frequent pixel values), one
for the front hand and one for the back hand, for each color channel
independently.

For the yellow channel we leave all the color range \[0 : 255\] without
limiting it to a certain range, this proved to be working better.
But for the other channels, we get the average value for the values got
from the front Crf, Cbf and back hand Crb, Cbb and put a
margin for tolerance.

![equation](http://www.sciweavers.org/upload/Tex2Img_1576958514/render.png)
![equation](http://www.sciweavers.org/upload/Tex2Img_1576958648/render.png)

Motion Detection
----------------

A motion mask is obtained by taking in consideration the two previous
frames and the difference between each one of them and the current frame
(`diff_1, diff_2`).

Then we apply a certain threshold >= 30 on the two difference
images and take the result of their AND-ing.
(diff\_1 >= 30) && (diff\_2 >= 30) This will results in a
noisy image, so we need to perform some Closing (Dilation then Erosion,
 10 operations) with an ellipse shaped structuring element in
order to fill the gapes between the hand shape and remove noisy
motions.

Then we get the largest contour in the resulted image and check that
this contour has a reasonable dimensions (area and aspect ratio) to be
accepted as a hand (area >= 2000 pixel, ![equation](http://www.sciweavers.org/upload/Tex2Img_1576959023/render.png) >= 1). In case of static
scenes and the motion mask is all zeros, we will take the last valid
contour as our current motion contour.

At the end, our motion mask will be the bounding rectangle containing
this largest valid contour.

Skin Detection
--------------

First, we remove the face by the ready Haar Cascade OpenCV face
detector. Then, we segment the image based on the color ranges we got in
the calibration step. So, any pixel having YCrCb value within our (min,
max) values will be equal to 1, otherwise it will be 0 in the skin mask.

Combining Results
-----------------

We will combine the motion and color mask by AND-ing them, after that we
will get the largest contour existing in the AND-ed image. The final
result will be a binary image having that contour filled with 1s and 0s
elsewhere.

Gesture Recognition
-------------------

In order to recognize and differentiate between gestures, we will use
geometry and the fact that different gestures have different areas with
spaces between fingers.
We only need the contour resulted from the combined image in the
previous step and the original frame in order to make our detection.

First, we specify a hyper-parameter $\epsilon$ which represents the
precision of the polygon containing the contour, then we get this
approximate polygon and the convex hull of this polygon.
Then, we get the area of the convex hull and the original contour to get
the area ratio between them according to the equations below, which we
will use next to differentiate between different gestures.
![equation](http://www.sciweavers.org/upload/Tex2Img_1576959223/render.png)

Then we get the convex hull of the approximated polygon and the defects
between the approximated polygon and its convex hull.
Defects consists of triangles between the hand fingers, a defect
consists of (start, end, far), which represent the vertices of the
triangle between two fingers. We get the lines between each two points
(i.e the triangle sides) to calculate the semi-perimeter and area.
Then, we get the height of the fingers according to the following
equation
 ![equation](http://www.sciweavers.org/upload/Tex2Img_1576959259/render.png) where
*lstart,end* is the base of the triangle (i.e distance between the
two fingers).

After that, we get the angle between the two fingers. If the angle is >= 90 & height > 30 then it is a valid defect and we can say that there is a triangle in this place.

We determine the gesture according to the number of defects which also
represents the number of raised fingers, the area of contour and the
area ratio that we calculated previously.

Experimental Results
====================

We were able to run the algorithm in real time with high frame rate. It
was also able to play some simple games in real time.

![image](https://user-images.githubusercontent.com/32196766/71313043-e9a81a80-243b-11ea-849e-c364964776dd.png)

Our approache proved to be working pretty well in case of a plain
background.

![image](https://user-images.githubusercontent.com/32196766/71313051-00e70800-243c-11ea-9281-ccc9b7d9af2e.png)

In normal backgrounds the hand detection was a little bit worse.

![image](https://user-images.githubusercontent.com/32196766/71313060-15c39b80-243c-11ea-93c5-8a8ff48c99c3.png)

Conclusion
==========

Traditional image processing techniques are powerful and more reliable
than deep learning techniques in many situations. It is always better to
have a working non-learning based application than a learning based
one.
Our approach proved to be working perfectly under some conditions. But,
it sometimes fails in case of noisy backgrounds.

But we still have a major plus which is operating in real time.

References
==========
[1] Ekaterini Stergiopoulou, Kyriakos Sgouropoulos, Nikos
Nikolaou, Nikos Papamarkos, Nikos Mitianoudis *Real time hand detection
in a complex background*. Engineering Applications of Artificial
Intelligence 35 (2014) 54–70
