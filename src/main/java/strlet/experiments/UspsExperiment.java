package strlet.experiments;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class UspsExperiment extends AbstractExperiment {

	private static final int FOLDS = 20;

	private static Instances mnist = null;
	private static Instances usps = null;
	private static Instances test = null;

	protected static synchronized void loadData() throws Exception {

		if (mnist != null)
			return;

		mnist = getMnistData("train-labels.idx1-ubyte",
				"train-images.idx3-ubyte");
		mnist.randomize(new Random(1));
		usps = getUsps("usps.arff");
		usps.randomize(new Random(1234));
		usps.stratify(FOLDS);
		test = getUsps("usps_test.arff");

	}

	private static Instances getMnistData(String labels, String images)
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

			Instances data = createInstances(rows * columns, numImages);

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
				Instance instance = new Instance(1, vals);
				data.add(instance);
			}
			return data;
		} finally {
			zip.close();
		}

	}

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

	private static Instances createInstances(int numPixels, int numImages) {

		FastVector attInfo = new FastVector(numPixels + 1);
		for (int i = 0; i < numPixels; ++i) {
			Attribute att = new Attribute("P" + i);
			attInfo.addElement(att);
		}

		FastVector classInfo = new FastVector(10);
		for (int i = 0; i < 10; ++i) {
			classInfo.addElement("" + i);
		}
		Attribute classAtt = new Attribute("class", classInfo);
		attInfo.addElement(classAtt);

		Instances empty = new Instances("MNIST", attInfo, numImages);
		empty.setClassIndex(empty.numAttributes() - 1);
		return empty;
	}

	private static Instances getUsps(String string) throws IOException {
		ZipFile uspsZip = new ZipFile(new File("testData" + File.separator
				+ "usps.zip"));
		ZipEntry entry = uspsZip.getEntry(string);
		ArffLoader a = new ArffLoader();
		a.setSource(uspsZip.getInputStream(entry));
		Instances test = a.getDataSet();
		test.setClassIndex(test.numAttributes() - 1);
		uspsZip.close();
		return test;
	}

	public void runExperiment(SingleSourceTransfer[] classifiers)
			throws Exception {

		UspsExperiment.loadData();
		double[] totals = new double[classifiers.length];
		for (int fold = 0; fold < FOLDS; ++fold) {
			Instances source = new Instances(mnist);
			Instances target = usps.testCV(FOLDS, fold);
			for (int i = 0; i < totals.length; ++i) {
				SingleSourceTransfer dup = classifiers[i].makeDuplicate();
				dup.buildModel(source, target);
				totals[i] += err(dup, test);
			}
		}

		for (int i = 0; i < totals.length; ++i) {
			System.out.println(ToPerc(totals[i] / FOLDS));
		}

	}
	
	public static void main(String[] args) throws Exception {
		UspsExperiment experiment = new UspsExperiment();
		experiment.runExperiment();
	}

}
