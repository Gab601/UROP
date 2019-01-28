import org.opencv.core.Core;
import org.opencv.core.Mat;
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


public class SkinColorTest {

    public static void main(String args[]) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat mat = new Mat();
        Image rawImage;
        int height = 0;
        int width = 0;


        BufferedImage finalImage = new BufferedImage(512, 512, 1);
        Color skinColor =  new Color(190, 130, 120);
        double skinTolerance = 0.1;
        for (int g = 0; g < 512; g++) {
            for (int b = 0; b < 512; b++) {
                double currRed = 255;
                double rgRatioDiff = Math.abs(2*currRed / g - (double) (skinColor.getRed()) / (double) (skinColor.getGreen()));
                double rbRatioDiff = Math.abs(2*currRed / b - (double) (skinColor.getRed()) / (double) (skinColor.getBlue()));
                if (Math.abs(rgRatioDiff - skinTolerance) < 0.01 || Math.abs(rbRatioDiff - skinTolerance) < 0.01) {
                    finalImage.setRGB(g, b, new Color(255, 255, 255).getRGB());
                }
                else {
                    finalImage.setRGB(g, b, new Color((int)currRed, g/2, b/2).getRGB());
                }
            }
        }
        DisplayImage(finalImage);
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
        BufferedImage blurredImage = new BufferedImage(width, height, 1);
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
                blurredImage.setRGB(w, h, new Color((int) color[0], (int) color[1], (int) color[2]).getRGB());
            }
        }
        return blurredImage;
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