import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;



public class HandDetection2 {

    public static void main(String args[]) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);


        /**
         * USER-MODIFIABLE VARIABLES
         **/

        int outlierTolerance = 110;
        int compressionFactor = 1;
        String videoPath = "/home/gkammer/Documents/UROP/london.m4v";
        Background background = Background.FINAL;
        Color skinColor = new Color(255, 200, 145);


        /**
         * END OF USER-MODIFIABLE VARIABLES
         **/

        int currentFrame = 0;
        int frame = 1090;
        VideoCapture videoCapture = new VideoCapture(videoPath);

        Mat originalMat = new Mat();
        int height = 0;
        int width = 0;

        while (currentFrame < frame) {
            if (videoCapture.read(originalMat)) {
                currentFrame++;
            } else {
                System.out.println("Unable to read from file!");
                break;
            }
        }
        if (videoCapture.read(originalMat)) {

            Mat hsvMat = new Mat();
            Mat bwMat = new Mat();
            Mat morphOutput = new Mat();

            Imgproc.cvtColor(originalMat, hsvMat, Imgproc.COLOR_BGR2HSV);

            CascadeClassifier faceDetector = new CascadeClassifier("/home/gkammer/Downloads/opencv-3.4.4/data/haarcascades/haarcascade_frontalface_defamatult.xml");
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(originalMat, faceDetections);
            for (Rect rect : faceDetections.toArray()) {
                Imgproc.rectangle(originalMat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255));
            }

            Mat skinHistogram = new Mat(256, 256, 16);
            int[] brightnesses = new int[256];
            if (faceDetections.toArray().length == 1) {
                skinHistogram = CreateSkinHistogram(hsvMat, faceDetections.toArray()[0]);
                brightnesses = CreateBrightnessDistribution(hsvMat, faceDetections.toArray()[0]);
                DisplayImage(Mat2Image(skinHistogram));
            }

            Scalar minValues = new Scalar(0, 0, 0);
            Scalar maxValues = new Scalar(40, 255, 255);
            Mat mask2 = new Mat();
            Core.inRange(hsvMat, minValues, maxValues, bwMat);
            DisplayImage(Mat2Image(bwMat));
            ApplyHistogramFilter(hsvMat, bwMat, skinHistogram, brightnesses);
            DisplayImage(Mat2Image(bwMat));
            EraseSmallBlobNoise(bwMat, 3);
            DisplayImage(Mat2Image(bwMat));
            /*
            int radius = 7;
            for (int r = radius; r < hsvMat.rows()-radius; r++) {
                for (int c = radius; c < hsvMat.cols()-radius; c++) {
                    int count = 0;
                    for (int dr = -radius; dr <= radius; dr++) {
                        for (int dc = -radius; dc <= radius; dc++) {
                            if (bwMat.get(r+dr, c + dc)[0] > 0) {
                                count++;
                            }
                        }
                    }
                    if (count > 110) {
                        mask2.put(r, c, 255);
                    }
                }
            }*/
            Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(50, 50));
            Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 1));
            //DisplayImage(Mat2Image(mask2));
            Imgproc.erode(mask2, morphOutput, erodeElement);
            //DisplayImage(Mat2Image(morphOutput));
            Imgproc.dilate(morphOutput, morphOutput, dilateElement);

            ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
            Mat hierarchy = new Mat();

            // if any contour exists...
            if (hierarchy.size().height > 0 && hierarchy.size().width > 0) {
                // for each contour, display it in blue
                for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0]) {
                    Imgproc.drawContours(originalMat, contours, idx, new Scalar(250, 0, 0));
                }
            }
            //DisplayImage(Mat2Image(originalMat));
        } else {
            System.out.println("Unable to read from file!");
        }
    }

    public static void EraseSmallBlobNoise(Mat originalMat, int blobsToKeep) {
        int[] largestSizes = new int[blobsToKeep+1];
        Point[] largestMats = new Point[blobsToKeep+1];
        for (int r = 0; r < originalMat.rows(); r++) {
            for (int c = 0; c < originalMat.cols(); c++) {
                if (originalMat.get(r, c)[0] > 1) {
                    largestSizes[blobsToKeep] = GetBlobSizeAndMark(originalMat, r, c);
                    largestMats[blobsToKeep] = new Point(r, c);
                    for (int i = blobsToKeep - 1; i >= 0; i--) {
                        if (largestSizes[i+1] > largestSizes[i]) {
                            int temp = largestSizes[i];
                            largestSizes[i] = largestSizes[i+1];
                            largestSizes[i+1] = temp;

                            Point tempPoint = largestMats[i];
                            largestMats[i] = largestMats[i+1];
                            largestMats[i+1] = tempPoint;
                        }
                    }
                }
            }
        }
        for (int r = 0; r < originalMat.rows(); r++) {
            for (int c = 0; c < originalMat.cols(); c++) {
                if (originalMat.get(r, c)[0] == 1) {
                    originalMat.put(r, c, new int[] {255, 255, 255});
                }
            }
        }
        for (Point point: largestMats) {
            GetBlobSizeAndMark(originalMat, (int)point.x, (int)point.y);
        }
        for (int r = 0; r < originalMat.rows(); r++) {
            for (int c = 0; c < originalMat.cols(); c++) {
                if (originalMat.get(r, c)[0] == 1) {
                    originalMat.put(r, c, new int[] {0, 0, 0});
                }
            }
        }
    }

    public static int GetBlobSizeAndMark(Mat inputMat, int r, int c) {
        if (inputMat.get(r, c)[0] == 0) {
            return 0;
        }
        boolean hasChanged = true;
        int radius = 1;
        int size = 0;
        while (hasChanged) {
            hasChanged = false;
            for (int x = Math.max(-radius, -r); x <= Math.min(radius, inputMat.rows()-r-1); x++) {
                for (int y = Math.max(-radius, -c); y <= Math.min(radius, inputMat.cols()-c-1); y++) {
                    int cur = (int)inputMat.get(r+x, c+y)[0];
                    int up = (int)inputMat.get(r+x-1, c+y)[0];
                    int down = (int)inputMat.get(r+x+1, c+y)[0];
                    int left = (int)inputMat.get(r+x, c+y-1)[0];
                    int right = (int)inputMat.get(r+x, c+y+1)[0];
                    if (cur > 1 && (up == 1 || down == 1 || left == 1 || right == 1)) {
                        inputMat.put(r+x, c+y, new int[] {1, 1, 1});
                        hasChanged = true;
                        size++;
                    }
                }
            }
            radius++;
        }
        return size;
    }

    public static void ApplyHistogramFilter(Mat hsvMat, Mat drawableMat, Mat histogram, int[] brightnesses) {
        for (int r = 0; r < hsvMat.rows(); r++) {
            for (int c = 0; c < hsvMat.cols(); c++) {
                double red = hsvMat.get(r, c)[2];
                double green = hsvMat.get(r, c)[1];
                double blue = hsvMat.get(r, c)[0];
                float[] hsbColor = Color.RGBtoHSB((int)red, (int)green, (int)blue, null);
                int hue = (int)(hsbColor[0]*255);
                int sat = (int)(hsbColor[1]*255);
                int val = (int)(hsbColor[2]*255);
                if (histogram.get(hue, sat)[0] == 0 || brightnesses[val] < 20) {
                    double[] color = new double[] {0,0,0};
                    drawableMat.put(r, c, color);
                }
            }
        }
    }

    public static Mat CreateSkinHistogram(Mat hsvMat, Rect rect) {
        Mat skinHistogram = new Mat(256, 256, 16);
        int lowY = rect.y;
        int lowX = rect.x;
        int highY = rect.y + rect.height;
        int highX = rect.x + rect.width;
        int avgY = (lowY + highY) / 2;
        int avgX = (lowX + highX) / 2;
        for (int r = lowY; r < highY; r++) {
            for (int c = lowX; c < highX; c++) {
                if (Math.sqrt(Math.pow(r-avgY, 2) + Math.pow(1.2*(c-avgX), 2)) < Math.min(avgX - lowX, avgY - lowY)) {
                    skinHistogram.put((int)hsvMat.get(r, c)[0], (int)hsvMat.get(r, c)[1], new double[]{255, 255, 255});
                    skinHistogram.put(Math.min((int)hsvMat.get(r, c)[0] + 1, 255), (int)hsvMat.get(r, c)[1], new double[]{255, 255, 255});
                    skinHistogram.put(Math.min((int)hsvMat.get(r, c)[0] - 1, 255), (int)hsvMat.get(r, c)[1], new double[]{255, 255, 255});
                    skinHistogram.put((int)hsvMat.get(r, c)[0], Math.min((int)hsvMat.get(r, c)[1] + 1, 255), new double[]{255, 255, 255});
                    skinHistogram.put((int)hsvMat.get(r, c)[0], Math.min((int)hsvMat.get(r, c)[1] - 1, 255), new double[]{255, 255, 255});
                }
            }
        }
        Mat dilateTemplate = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
        Mat erodeTemplate = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8, 8));
        Imgproc.dilate(skinHistogram, skinHistogram, dilateTemplate);
        Imgproc.erode(skinHistogram, skinHistogram, erodeTemplate);
        for (int i = 0; i < 6; i++) {
            Imgproc.dilate(skinHistogram, skinHistogram, dilateTemplate);
        }
        return skinHistogram;
    }

    public static int[] CreateBrightnessDistribution(Mat hsvMat, Rect rect) {
        int[] brightnesses = new int[256];
        int lowY = rect.y;
        int lowX = rect.x;
        int highY = rect.y + rect.height;
        int highX = rect.x + rect.width;
        int avgY = (lowY + highY) / 2;
        int avgX = (lowX + highX) / 2;
        for (int r = lowY; r < highY; r++) {
            for (int c = lowX; c < highX; c++) {
                if (Math.sqrt(Math.pow(r-avgY, 2) + Math.pow(1.2*(c-avgX), 2)) < Math.min(avgX - lowX, avgY - lowY)) {
                    brightnesses[(int)hsvMat.get(r, c)[2]] += 1;
                }
            }
        }
        return brightnesses;
    }

    public static BufferedImage Mat2Image(Mat m) {
        // Fastest code
        // output can be assigned either to a BufferedImage or to an Image

        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (m.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels() * m.cols() * m.rows();
        byte[] b = new byte[bufferSize];
        m.get(0, 0, b); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    public static Mat Image2Mat(BufferedImage im) {
        // Convert INT to BYTE
        //im = new BufferedImage(im.getWidth(), im.getHeight(),BufferedImage.TYPE_3BYTE_BGR);
        // Convert bufferedimage to byte array
        byte[] pixels = ((DataBufferByte) im.getRaster().getDataBuffer())
                .getData();

        // Create a Matrix the same size of image
        Mat image = new Mat(im.getHeight(), im.getWidth(), CvType.CV_8UC3);
        // Fill Matrix with image values
        image.put(0, 0, pixels);

        return image;

    }

    public static void DisplayImage(BufferedImage bufferedImage) {
        ImageIcon imageIcon;
        JLabel label;
        JFrame frame;
        imageIcon = new ImageIcon(bufferedImage);
        label = new JLabel(imageIcon);
        frame = new JFrame();
        frame.setLayout(new FlowLayout());
        frame.setSize(bufferedImage.getWidth(null) + 50, bufferedImage.getHeight(null) + 50);
        frame.add(label);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public static BufferedImage FlipImage(BufferedImage bufferedImage) {
        int width = bufferedImage.getWidth();
        int height = bufferedImage.getHeight();
        BufferedImage flippedImage = new BufferedImage(width, height, 1);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                flippedImage.setRGB(w, h, ((BufferedImage) bufferedImage).getRGB(width - w - 1, height - h - 1));
            }
        }
        return flippedImage;
    }

    public static int GetValue(ArrayList<BufferedImage> arrayList, int frame, int width, int height, int color) {
        if (color == 0) {
            return new Color(arrayList.get(frame).getRGB(width, height)).getRed();
        } else if (color == 1) {
            return new Color(arrayList.get(frame).getRGB(width, height)).getGreen();
        } else {
            return new Color(arrayList.get(frame).getRGB(width, height)).getBlue();
        }
    }

    public static BufferedImage CompressImage(BufferedImage bufferedImage, int factor) {
        BufferedImage outImage = new BufferedImage(bufferedImage.getWidth() / factor, bufferedImage.getHeight() / factor, 1);
        for (int h = 0; h < outImage.getHeight(); h++) {
            for (int w = 0; w < outImage.getWidth(); w++) {
                outImage.setRGB(w, h, bufferedImage.getRGB(factor * w, factor * h));
            }
        }
        return outImage;
    }

    public static BufferedImage BlurImage(BufferedImage bufferedImage) {
        int width = bufferedImage.getWidth();
        int height = bufferedImage.getHeight();
        BufferedImage mat = new BufferedImage(width, height, 1);
        for (int h = 1; h < height - 1; h++) {
            for (int w = 1; w < width - 1; w++) {
                double[] color = new double[3];
                double[][] weights = new double[][]{
                        {0.05, 0.15, 0.05},
                        {0.15, 0.2, 0.15},
                        {0.05, 0.15, 0.05}};
                for (int a = -1; a <= 1; a++) {
                    for (int b = -1; b <= 1; b++) {
                        color[0] += new Color(bufferedImage.getRGB(w + a, h + b)).getRed() * weights[a + 1][b + 1];
                        color[1] += new Color(bufferedImage.getRGB(w + a, h + b)).getGreen() * weights[a + 1][b + 1];
                        color[2] += new Color(bufferedImage.getRGB(w + a, h + b)).getBlue() * weights[a + 1][b + 1];
                    }
                }
                mat.setRGB(w, h, new Color((int) color[0], (int) color[1], (int) color[2]).getRGB());
            }
        }
        return mat;
    }

    public static BufferedImage CreateMovementImage(ArrayList<BufferedImage> videoArrayList, int outlierTolerance, Color skinColor, double skinTolerance, Background bkGround) {
        int numFrames = videoArrayList.size();
        int width = videoArrayList.get(0).getWidth();
        int height = videoArrayList.get(0).getHeight();
        BufferedImage finalImage = new BufferedImage(width, height, 1);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                double avgRed = 0;
                double avgGreen = 0;
                double avgBlue = 0;
                for (int i = 0; i < numFrames; i++) {
                    avgRed += (double) GetValue(videoArrayList, i, w, h, 0) / (double) numFrames;
                    avgGreen += (double) GetValue(videoArrayList, i, w, h, 1) / (double) numFrames;
                    avgBlue += (double) GetValue(videoArrayList, i, w, h, 2) / (double) numFrames;
                }
                int outlierSumRed = 0;
                int outlierSumGreen = 0;
                int outlierSumBlue = 0;
                int outlierNum = 0;
                for (int i = 0; i < numFrames; i++) {
                    double currRed = GetValue(videoArrayList, i, w, h, 0);
                    double currGreen = GetValue(videoArrayList, i, w, h, 1);
                    double currBlue = GetValue(videoArrayList, i, w, h, 2);
                    double redOutlier = Math.abs(currRed - avgRed);
                    double greenOutlier = Math.abs(currGreen - avgGreen);
                    double blueOutlier = Math.abs(currBlue - avgBlue);

                    double rgRatioDiff = Math.abs(currRed / currGreen - (double) (skinColor.getRed()) / (double) (skinColor.getGreen()));
                    double rbRatioDiff = Math.abs(currRed / currBlue - (double) (skinColor.getRed()) / (double) (skinColor.getBlue()));

                    double outlierDistance = Math.pow(redOutlier, 2) + Math.pow(greenOutlier, 2) + Math.pow(blueOutlier, 2);
                    if (outlierDistance > outlierTolerance && rgRatioDiff < skinTolerance && rbRatioDiff < skinTolerance && (h > 0.34 * height || Math.abs(w - width / 2) > 0.25 * width) && Math.abs(currRed - skinColor.getRed()) < 60) {
                        if (numFrames - 1 == i) {
                            outlierSumRed = 0; //GetValue(videoArrayList, i, w, h, 0) * outlierNum;
                            outlierSumGreen = 0; //GetValue(videoArrayList, i, w, h, 1) * outlierNum;
                            outlierSumBlue = 0; //GetValue(videoArrayList, i, w, h, 2) * outlierNum;
                        }

                        outlierSumRed += GetValue(videoArrayList, i, w, h, 0);
                        outlierSumGreen += GetValue(videoArrayList, i, w, h, 1);
                        outlierSumBlue += GetValue(videoArrayList, i, w, h, 2);
                        outlierSumBlue += (double) (numFrames - i) / (double) numFrames * 200;
                        outlierSumRed += (double) i / (double) numFrames * 200;
                        outlierNum++;
                    }
                }
                Color color;
                if (outlierNum != 0) {
                    color = new Color(Math.min(255, outlierSumRed / outlierNum), Math.min(outlierSumGreen / outlierNum, 255), Math.min(outlierSumBlue / outlierNum, 255));
                } else {
                    switch (bkGround) {
                        case BLACK:
                            color = new Color(0, 0, 0);
                            break;
                        case WHITE:
                            color = new Color(255, 255, 255);
                            break;
                        case INITIAL:
                            color = new Color(GetValue(videoArrayList, 0, w, h, 0), GetValue(videoArrayList, 0, w, h, 1), GetValue(videoArrayList, 0, w, h, 2));
                            break;
                        case AVERAGE:
                            color = new Color((int) avgRed, (int) avgGreen, (int) avgBlue);
                            break;
                        default:
                            color = new Color(GetValue(videoArrayList, numFrames - 1, w, h, 0), GetValue(videoArrayList, numFrames - 1, w, h, 1), GetValue(videoArrayList, numFrames - 1, w, h, 2));
                    }
                }
                finalImage.setRGB(w, h, color.getRGB());
            }
        }
        return finalImage;
    }

    public static ArrayList<double[]> CreateStrokeArrayFromFile(String filename) {
        ArrayList<double[]> strokeTimes = new ArrayList<>();
        BufferedReader br = null;
        String line = "";
        String cvsSplitBy = ",";
        int iter = 0;
        try {
            br = new BufferedReader(new FileReader(filename));
            while ((line = br.readLine()) != null) {

                // use comma as separator
                String[] stroke = line.split(cvsSplitBy);
                if (iter > 0) {
                    strokeTimes.add(new double[]{Double.parseDouble(stroke[0]), Double.parseDouble(stroke[1])});
                } else {
                    iter++;
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return strokeTimes;
    }

    public enum Background {BLACK, WHITE, AVERAGE, INITIAL, FINAL}
}