package com.novemser;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.zip.ZipFile;

public class ReadHelper {
  private static int readInt(InputStream labelFile) throws IOException {
    byte[] number = new byte[4];
    labelFile.read(number);
    int retVal = 0;
    for (int i = 0; i < 4; ++i) {
      retVal *= 256;
      if (number[i] < 0)
        retVal += (256 + number[i]);
      else
        retVal += number[i];
    }
    return retVal;
  }

  public static void main(String[] args) throws IOException {
    double[][] data = ReadHelper.getMnistData("train-labels.idx1-ubyte",
        "train-images.idx3-ubyte");
    CSVPrinter printer = new CSVPrinter(new FileWriter("/home/novemser/mnist.csv"),
        CSVFormat.DEFAULT);

    System.out.println(data.length);
    System.out.println(data[0].length - 1);
    for (double[] line : data) {
      for (double v : line) {
        printer.print(v);
      }
      printer.println();
//      System.out.println(Arrays.toString(line));
    }
    printer.close();
  }

  public static double[][] getMnistData(String labels, String images)
      throws IOException {

    ZipFile zip = new ZipFile(new File("testData" + File.separator
        + "MNIST.zip"));

    try {
      InputStream labelFile = zip.getInputStream(zip.getEntry(labels));

      if (readInt(labelFile) != 2049) {
        System.err.println("Corrupt label file");
        System.exit(1);
      }
      int numImages = readInt(labelFile);

      InputStream imagesFile = zip.getInputStream(zip.getEntry(images));
      if (readInt(imagesFile) != 2051) {
        System.err.println("Corrupt images file");
        System.exit(1);
      }

      if (readInt(imagesFile) != numImages) {
        System.err
            .println("Number of images different than number of labels");
        System.exit(1);
      }

      int rows = readInt(imagesFile);
      int columns = readInt(imagesFile);

      double[][] data = new double[numImages][];

      byte[] line = new byte[rows * columns];
      for (int image = 0; image < numImages; ++image) {
        int label = labelFile.read();
        int off = 0;
        int len = line.length;
        while (len > 0) {
          int tmp = imagesFile.read(line, off, len);
          off += tmp;
          len -= tmp;
        }

        double[] vals = new double[line.length + 1];
        for (int i = 0; i < line.length; ++i) {
          vals[i] = (line[i] & 0xFF);
        }
        vals[line.length] = label;
        data[image] = vals;
      }
      return data;
    } finally {
      zip.close();
    }
  }
}
