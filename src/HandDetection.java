import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.ArrayList;


public class HandDetection {

    /**
     * USER-MODIFIABLE VARIABLES
     **/

    public static int histogramCompression = 8;
    public static int compressionRate = 2;
    public static String videoPath = "/home/gkammer/Documents/UROP/london.m4v";
    public static String timeChartLocation = "/home/gkammer/Documents/UROP/london_test.csv";
    public static String cascadeClassifierLocation = "/home/gkammer/Downloads/opencv-3.4.4/data/haarcascades/haarcascade_frontalface_defamatult.xml";
    public static String savePath = "/home/gkammer/Documents/UROP/final";
    public static String tempSavePath = "/home/gkammer/Documents/UROP/temp";


    /**
     * END OF USER-MODIFIABLE VARIABLES
     **/

    public static int histogramSize = 256/histogramCompression;
    public static int currentFrame = 0;
    public static File tempFile = new File(tempSavePath);

    public static void main(String args[]) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        VideoCapture videoCapture = new VideoCapture(videoPath);

        ArrayList<double[]> strokeFrames = CreateStrokeArrayFromFile(timeChartLocation);
        for (double[] i: strokeFrames) {
            i[0] = (int)(29.97*i[0]);
            i[1] = (int)(29.97*i[1]);
        } //change from seconds to frames

        Mat rawMat = new Mat();
        Mat finalMat = new Mat();

        if (videoCapture.read(rawMat)) {
            finalMat = CompressMat(rawMat, compressionRate);
            currentFrame++;
        }
        else {
            System.out.println("Unable to read from file!");
        }

        for (double[] times: strokeFrames) {
            System.out.println("\n\nStarting new stroke");
            System.out.println("Stroke time: " + Math.round((times[0]/29.97)*10)/10.0 + " to " + Math.round((times[1]/29.97)*10)/10.0 + " seconds (frames " + (int)times[0] + " to " + (int)times[1] + ")");
            System.out.println("Total number of frames in stroke: " + (int)(times[1]-times[0]));
            CreateStrokeImage(times, finalMat, videoCapture);
            DisplayMat(finalMat);
            try {
                File outputfile = new File(savePath + "_" + times[0] + "-" + times[1]);
                ImageIO.write(Mat2Image(finalMat), "png", outputfile);
            } //save the image
            catch (IOException e) { }
        }
    }
    
    public static void CreateStrokeImage(double[] times, Mat finalMat, VideoCapture videoCapture) {
        Mat rawMat = new Mat();
        Mat[] originalMats = new Mat[(int)(times[1]-times[0])];
        Mat[] finalBwMats = new Mat[(int)(times[1]-times[0])];
        Mat[] skinColorMats = new Mat[(int)(times[1]-times[0])];

        for (int r = 0; r < finalMat.rows(); r++) {
            for (int c = 0; c < finalMat.cols(); c++) {
                finalMat.put(r, c, new double[] {0, 0, 0});
            }
        } //set finalMat to all black

        while (currentFrame < times[0]) {
            if (videoCapture.read(rawMat)) {
                currentFrame++;
            } else {
                System.out.println("Unable to read from file!");
                break;
            }
        } //Get to the first used frame

        while (currentFrame < times[1]) {
            int frameNumber = currentFrame-(int)times[0];
            currentFrame++;
            System.out.print("\rCurrent frame: " + (frameNumber+1));
            System.out.flush();
            if (videoCapture.read(rawMat)) {
                originalMats[frameNumber] = CompressMat(rawMat, compressionRate);
                Mat hsvMat = new Mat();
                Mat skinColorMat = new Mat();
                Mat bwMat = new Mat();

                try { ImageIO.write(Mat2Image(originalMats[frameNumber]), "png", tempFile); } catch (Exception e) { }
                BodyDetection.RemoveBackground(tempSavePath, tempSavePath);

                BufferedImage tempBuff = null;
                try { tempBuff = ImageIO.read(tempFile); } catch (Exception e) { }
                originalMats[frameNumber] = Image2Mat(tempBuff);

                Imgproc.cvtColor(originalMats[frameNumber], hsvMat, Imgproc.COLOR_BGR2HSV);
                MatOfRect faceDetections = new MatOfRect();
                CascadeClassifier faceDetector = new CascadeClassifier(cascadeClassifierLocation);
                faceDetector.detectMultiScale(originalMats[frameNumber], faceDetections, 1.05, 5);

                //DisplayMat(originalMats[frameNumber]);
                boolean[][][] skinHistogram;
                if (faceDetections.toArray().length >= 1) {
                    skinHistogram = CreateSkinHistogram(hsvMat, faceDetections.toArray()[0]);
                    Core.inRange(hsvMat, new Scalar(0, 1, 1), new Scalar(40, 254, 254), skinColorMat);
                    Core.inRange(hsvMat, new Scalar(0, 1, 1), new Scalar(40, 254, 254), bwMat);

                    ApplyHistogramFilter(hsvMat, bwMat, skinHistogram);

                    EraseSmallBlobNoise(bwMat, originalMats[frameNumber], faceDetections.toArray()[0], 3);

                    Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(20, 20));
                    Imgproc.dilate(bwMat, bwMat, dilateElement);
                }
                else {
                    System.out.println(" - No face detected");
                    Core.inRange(hsvMat, new Scalar(0, 1, 1), new Scalar(40, 254, 254), skinColorMat);
                    Core.inRange(hsvMat, new Scalar(0, 1, 1), new Scalar(40, 254, 254), bwMat);

                    EraseSmallBlobNoise(bwMat, originalMats[frameNumber], 3);
                }
                finalBwMats[frameNumber] = bwMat;
                skinColorMats[frameNumber] = skinColorMat;
            } else {
                System.out.println(" - Unable to read from file!");
            }
        }
        int[] sumColorNonHand = new int[3];
        int numColorNonHand;
        Mat avgMat = new Mat(finalMat.rows(), finalMat.cols(), 16);
        for (int r = 0; r < finalMat.rows(); r++) {
            for (int c = 0; c < finalMat.cols(); c++) {
                sumColorNonHand[0] = 0;
                sumColorNonHand[1] = 0;
                sumColorNonHand[2] = 0;
                numColorNonHand = 0;
                for (int f = 0; f < (times[1]-times[0]); f++) {
                    if (finalBwMats[f].get(r, c)[0] == 0 && originalMats[f].get(r, c)[0] > 0) {
                        sumColorNonHand[0] += originalMats[f].get(r, c)[0];
                        sumColorNonHand[1] += originalMats[f].get(r, c)[1];
                        sumColorNonHand[2] += originalMats[f].get(r, c)[2];
                        numColorNonHand++;
                    }
                }
                if (numColorNonHand == 0) {
                    avgMat.put(r, c, new double[] {0, 0, 0});
                }
                else {
                    avgMat.put(r, c, new double[] {sumColorNonHand[0]/numColorNonHand, sumColorNonHand[1]/numColorNonHand, sumColorNonHand[2]/numColorNonHand});
                }
                for (int f = 0; f < (times[1]-times[0]); f++) {
                    if (finalBwMats[f].get(r, c)[0] > 0 && skinColorMats[f].get(r, c)[0] > 0) {
                        if (avgMat.get(r, c)[0] == 0 || GetPixelDifference(avgMat.get(r, c), originalMats[f].get(r, c)) > 40) {
                            finalMat.put(r, c, new double[]{255 * ((times[1]-times[0]) - f) / (times[1] - times[0]), 0, 255 * f / (times[1] - times[0])});
                        }
                    }
                }
            }
        } //create avgMat which is the average of all non-hand pixels over the timeframe of the stroke, use those values to shade in detected hands only if they are out of the ordinary in that pixel location
        DisplayMat(avgMat);

    }


    /**
    Gets rid of all but the largest blobs in a black/white image, number is specified by user
    All chosen blobs must have a similar color profile as the blobs within the rectangle
     **/
    public static void EraseSmallBlobNoise(Mat bwMat, Mat originalMat, Rect rect, int blobsToKeep) {
        int lowY = rect.y;
        int lowX = rect.x;
        int highY = rect.y + rect.height;
        int highX = rect.x + rect.width;
        int faceSize = 1;
        int[] sumRGB = new int[] {0, 0, 0};
        for (int r = lowY; r < highY; r++) {
            for (int c = lowX; c < highX; c++) {
                if (bwMat.get(r, c)[0] > 0) {
                    sumRGB[0] += originalMat.get(r, c)[0];
                    sumRGB[1] += originalMat.get(r, c)[1];
                    sumRGB[2] += originalMat.get(r, c)[2];
                    faceSize++;
                }
            }
        }
        double[] avgRGB = new double[] {sumRGB[0]/faceSize, sumRGB[1]/faceSize, sumRGB[2]/faceSize};
        Blob largestBlobs[] = new Blob[blobsToKeep+1];
        for (int i = blobsToKeep - 1; i >= 0; i--) {
            largestBlobs[i] = new Blob();
        }
        for (int r = 0; r < bwMat.rows(); r++) {
            for (int c = 0; c < bwMat.cols(); c++) {
                if (bwMat.get(r, c)[0] > 128) {
                    largestBlobs[blobsToKeep] = new Blob(bwMat, originalMat, r, c);
                    double totalDiff = GetPixelDifference(largestBlobs[blobsToKeep].avgRGB, avgRGB);
                    if (totalDiff < 40) {
                        largestBlobs[blobsToKeep].shadeBlob(1);
                        for (int i = blobsToKeep - 1; i >= 0; i--) {
                            if (largestBlobs[i+1].size > largestBlobs[i].size) {
                                Blob temp = largestBlobs[i];
                                largestBlobs[i] = largestBlobs[i+1];
                                largestBlobs[i+1] = temp;
                            }
                        }
                    }
                }
            }
        }
        //DisplayMat(bwMat);
        for (int i = 0; i < largestBlobs.length-1; i++) {
            if (largestBlobs[i].size > 0) {
                largestBlobs[i].shadeBlob(255);
            }
        }
        for (int r = 0; r < bwMat.rows(); r++) {
            for (int c = 0; c < bwMat.cols(); c++) {
                if (bwMat.get(r, c)[0] <= 128) {
                    bwMat.put(r, c, new double[] {0}); //reset unused pixel values
                }
            }
        }
        //DisplayMat(bwMat);
    }

    public static void EraseSmallBlobNoise(Mat bwMat, Mat originalMat, int blobsToKeep) {
        Blob largestBlobs[] = new Blob[blobsToKeep+1];
        for (int i = blobsToKeep - 1; i >= 0; i--) {
            largestBlobs[i] = new Blob();
        }
        for (int r = 0; r < bwMat.rows(); r++) {
            for (int c = 0; c < bwMat.cols(); c++) {
                if (bwMat.get(r, c)[0] > 128) {
                    largestBlobs[blobsToKeep] = new Blob(bwMat, originalMat, r, c);
                    largestBlobs[blobsToKeep].shadeBlob(1);
                    for (int i = blobsToKeep - 1; i >= 0; i--) {
                        if (largestBlobs[i+1].size > largestBlobs[i].size) {
                            Blob temp = largestBlobs[i];
                            largestBlobs[i] = largestBlobs[i+1];
                            largestBlobs[i+1] = temp;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < largestBlobs.length-1; i++) {
            if (largestBlobs[i].size > 0) {
                largestBlobs[i].shadeBlob(255);
            }
        }
        for (int r = 0; r < bwMat.rows(); r++) {
            for (int c = 0; c < bwMat.cols(); c++) {
                if (bwMat.get(r, c)[0] <= 128) {
                    bwMat.put(r, c, new double[] {0}); //reset unused pixel values
                }
            }
        }
    }

    /**
    Given a histogram of acceptable HSV skin values, make all pixels outside of those values black in drawableMat
     */
    public static void ApplyHistogramFilter(Mat hsvMat, Mat drawableMat, boolean[][][] histogram) {
        for (int r = 0; r < hsvMat.rows(); r++) {
            for (int c = 0; c < hsvMat.cols(); c++) {
                int hue = (int)hsvMat.get(r, c)[0];
                int sat = (int)hsvMat.get(r, c)[1];
                int val = (int)hsvMat.get(r, c)[2];
                if (histogram[hue/histogramCompression][sat/histogramCompression][val/histogramCompression] == false) {
                    double[] color = new double[] {0};
                    drawableMat.put(r, c, color);
                }
            }
        }
    }

    public static boolean[][][] CreateSkinHistogram(Mat hsvMat, Rect rect) {
        int[][][] skinHistogram = new int[histogramSize][histogramSize][histogramSize];
        boolean[][][] boolHistogram = new boolean[histogramSize][histogramSize][histogramSize];
        int lowY = rect.y;
        int lowX = rect.x;
        int highY = rect.y + rect.height;
        int highX = rect.x + rect.width;
        int avgY = (lowY + highY) / 2;
        int avgX = (lowX + highX) / 2;
        for (int r = lowY; r < highY; r++) {
            for (int c = lowX; c < highX; c++) {
                if (Math.sqrt(Math.pow(1.2*(r-avgY), 2) + Math.pow(1.3*(c-avgX), 2)) < Math.min(avgX - lowX, avgY - lowY)) {
                    double[] color = hsvMat.get(r, c);
                    skinHistogram[(int)color[0]/histogramCompression][(int)color[1]/histogramCompression][(int)color[2]/histogramCompression] += 1;
                    boolHistogram[(int)color[0]/histogramCompression][(int)color[1]/histogramCompression][(int)color[2]/histogramCompression] = true;
                }
            }
        }
        //DisplayMat(hsvMat);
        for (int i = 0; i < histogramSize; i++) {
            for (int j = 0; j < histogramSize; j++) {
                for (int k = 0; k < histogramSize; k++) {
                    if (skinHistogram[i][j][k] > 0) {
                        int surrounded = 0;
                        surrounded += skinHistogram[Math.max(i - 1, 0)][j][k];
                        surrounded += skinHistogram[Math.min(i + 1, histogramSize-1)][j][k];
                        surrounded += skinHistogram[i][Math.max(j - 1, 0)][k];
                        surrounded += skinHistogram[i][Math.min(j + 1, histogramSize-1)][k];
                        surrounded += skinHistogram[i][j][Math.max(k - 1, 0)];
                        surrounded += skinHistogram[i][j][Math.min(k + 1, histogramSize-1)];
                        if (surrounded < (rect.height*rect.width)/200) {
                            boolHistogram[i][j][k] = false;
                        }
                    }
                }
            }
        }
        return boolHistogram;
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

    public static Mat Image2Mat(BufferedImage bufferedImage) {
        Mat mat = new Mat(bufferedImage.getHeight(), bufferedImage.getWidth(), 16);
        for (int r = 0; r < bufferedImage.getWidth(); r++) {
            for (int c = 0; c < bufferedImage.getHeight(); c++) {
                Color color = new Color(bufferedImage.getRGB(r, c));
                if ((bufferedImage.getRGB(r, c)>>24) == 0x00) {
                    mat.put(c, r, new double[] {0, 0, 0});
                } else {
                    mat.put(c, r, new double[] {color.getBlue(), color.getGreen(), color.getRed()});
                }

            }
        }
        return mat;
    }

    public static void DisplayMat(Mat mat) {
        DisplayImage(Mat2Image(mat));
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

    public static Mat CompressMat(Mat mat, int factor) {
        Mat outMat = new Mat(mat.rows() / factor, mat.cols() / factor, mat.type());
        for (int c = 0; c < outMat.cols(); c++) {
            for (int r = 0; r < outMat.rows(); r++) {
                outMat.put(r, c, mat.get(factor * r, factor * c));
            }
        }
        return outMat;
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

    public static void SeparateEdges(Mat bwMat, Mat originalMat, int t1, int t2, int a) {
        Mat edges = new Mat();
        Imgproc.Canny(originalMat, edges, t1, t2, a);
        //DisplayMat(edges);
        for (int r = 0; r < bwMat.rows(); r++) {
            for (int c = 0; c < bwMat.cols(); c++) {
                if (edges.get(r, c)[0] > 0) {
                    bwMat.put(r, c, new double[] {0});
                }
            }
        }
    }

    public static double GetPixelDifference(double[] p1, double[] p2) {
        double diffRed = Math.abs(p1[0] - p2[0]);
        double diffGreen = Math.abs(p1[1] - p2[1]);
        double diffBlue = Math.abs(p1[2] - p2[2]);
        return Math.sqrt(Math.pow(diffRed, 2) + Math.pow(diffGreen, 2) + Math.pow(diffBlue, 2));
    }

public enum Background {BLACK, WHITE, AVERAGE, INITIAL, FINAL}
}