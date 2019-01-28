import org.opencv.core.Mat;

public class Blob {
    public int size;
    public double[] avgRGB;
    public Mat bwMat;
    public Mat rgbMat;
    public int r;
    public int c;

    public Blob() {
        this.size = 0;
        this.avgRGB = new double[] {0,0,0};
        this.bwMat = new Mat();
        this.rgbMat = new Mat();
        this.r = 0;
        this.c = 0;
    }

    public Blob(Mat bwMat, Mat rgbMat, int r, int c) {
        this.bwMat = bwMat;
        this.rgbMat = rgbMat;
        this.r = r;
        this.c = c;
        if (bwMat.get(r, c)[0] == 0) {
            size = 0;
            avgRGB = new double[] {0, 0, 0};
        }
        this.bwMat.put(r, c, new double[] {128});
        boolean hasChanged = true;
        int radius = 1;
        this.size = 1;
        int[] sumRGB = new int[] {0, 0, 0};
        while (hasChanged) {
            hasChanged = false;
            for (int x = Math.max(-radius, -r); x <= Math.min(radius, bwMat.rows()-r-2); x++) {
                for (int y = Math.max(-radius, -c); y <= Math.min(radius, bwMat.cols()-c-2); y++) {
                    int cur = (int)this.bwMat.get(r+x, c+y)[0];
                    int up = (int)this.bwMat.get(r+x-1, c+y)[0];
                    int down = (int)this.bwMat.get(r+x+1, c+y)[0];
                    int left = (int)this.bwMat.get(r+x, c+y-1)[0];
                    int right = (int)this.bwMat.get(r+x, c+y+1)[0];
                    if (cur > 128 && (up == 128 || down == 128 || left == 128 || right == 128)) {
                        this.bwMat.put(r+x, c+y, new double[] {128});
                        sumRGB[0] += rgbMat.get(r+x, c+y)[0];
                        sumRGB[1] += rgbMat.get(r+x, c+y)[1];
                        sumRGB[2] += rgbMat.get(r+x, c+y)[2];
                        hasChanged = true;
                        this.size++;
                    }
                }
            }
            radius++;
        }
        if (this.size > 200) {
            //HandDetection.DisplayMat(bwMat);
        }
        this.avgRGB = new double[] {sumRGB[0]/size, sumRGB[1]/size, sumRGB[2]/size};
    }

    public void shadeBlob(double newColor) {
        double oldColor = this.bwMat.get(r, c)[0];
        this.bwMat.put(r, c, new double[] {newColor});
        boolean hasChanged = true;
        int radius = 1;
        while (hasChanged) {
            hasChanged = false;
            for (int x = Math.max(-radius, -r); x <= Math.min(radius, bwMat.rows()-r-2); x++) {
                for (int y = Math.max(-radius, -c); y <= Math.min(radius, bwMat.cols()-c-2); y++) {
                    int cur = (int)bwMat.get(r+x, c+y)[0];
                    int up = (int)bwMat.get(r+x-1, c+y)[0];
                    int down = (int)bwMat.get(r+x+1, c+y)[0];
                    int left = (int)bwMat.get(r+x, c+y-1)[0];
                    int right = (int)bwMat.get(r+x, c+y+1)[0];
                    if (cur == oldColor && (up == newColor || down == newColor || left == newColor || right == newColor)) {
                        bwMat.put(r+x, c+y, new double[] {newColor});
                        hasChanged = true;
                    }
                }
            }
            radius++;
        }
    }
}
