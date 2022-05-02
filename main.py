from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import dlib
import imageio
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './'

def applyAffineTransform(src, srcTri, dstTri, size) :
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return dst
def extract_index_nparray(nparray):
    index =None
    for num in nparray[0]:
        index = num
        break
    return index
def morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha) :
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    t1Rect = []
    t2Rect = []
    tRect = []
    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)#
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
    imgMorph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = imgMorph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


@app.route('/')
def index():
    return render_template("index.html",data="hey")

@app.route("/prediction",methods=["POST"])
def prediction():
    #img.save("img1.jpg")
    tempi = request.files['img1']
    tempi2 = request.files["img2"]
    path1 = os.path.join(app.config['UPLOAD_FOLDER'],tempi.filename)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'],tempi2.filename)
    tempi.save(path1)
    tempi2.save(path2)
    i = cv2.imread(path1)
    i2 = cv2.imread(path2)
    img = cv2.resize(i,(300,400))
    img2 = cv2.resize(i2,(300,400)) #background img
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    landmarks_points = []
    landmarks_points2 = []
    indexes_triangles = []
    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray,face)
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x,y))
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)

        #delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles,dtype=np.int32)

        for t in triangles:
            pt1 = (t[0],t[1])
            pt2 = (t[2],t[3])
            pt3 = (t[4],t[5])
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1,index_pt2,index_pt3]
                indexes_triangles.append(triangle)
    
    #second face
    faces2 = detector(img2_gray)
    for face in faces2:
        landmarks = predictor(img2_gray,face)
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x,y))

    for triangle_index in indexes_triangles:
        pt1 = landmarks_points2[triangle_index[0]]
        pt2 = landmarks_points2[triangle_index[1]]
        pt3 = landmarks_points2[triangle_index[2]]

    temp = 1
    images = []
    for i in range(1,5):
        images.append(img)
    while temp<=30:
        print(temp)
        alpha = temp/30
        points = []
        # NB: NOT CONVERT IN MAT TO FLOAT DATA TYPE
        for i in range(0, len(landmarks_points)):
            x = ( 1 - alpha ) * landmarks_points[i][0] + alpha * landmarks_points2[i][0]
            y = ( 1 - alpha ) * landmarks_points[i][1] + alpha * landmarks_points2[i][1]
            points.append((x,y))
        # Allocate space for final output
        imgMorph = np.zeros(img.shape, dtype = img.dtype)
        for triangle_index in indexes_triangles:
            x = triangle_index[0]
            y = triangle_index[1]
            z = triangle_index[2]
            x = int(x)
            y = int(y)
            z = int(z)
            pt1 = landmarks_points2[x]
            pt2 = landmarks_points2[y]
            pt3 = landmarks_points2[z]
            pt4 = landmarks_points[x]
            pt5 = landmarks_points[y]
            pt6 = landmarks_points[z]
            t1 = [pt4,pt5,pt6]
            t2 = [pt1,pt2,pt3]
            t = [ points[x], points[y], points[z] ]
            morphTriangle(img, img2, imgMorph, t1, t2, t, alpha)
        images.append(np.uint8(imgMorph))
        temp += 1
    for i in range(1,10):
        images.append(img2)
            
    os.remove(path1)
    os.remove(path2)
    pathl = os.path.join(app.config['UPLOAD_FOLDER'],"static/html.gif")
    with imageio.get_writer(pathl,mode="I") as writer:
        for image in images:
            rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_image)
    filename = "html.gif"
    return render_template("result.html",data=filename)

@app.route('/post', methods=["POST"])
def testpost():
     input_json = request.get_json(force=True) 
     dictToReturn = {'text':input_json['text']}
     return jsonify(dictToReturn)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0",port=port)

