package strlet.experiments;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.Random;
import java.util.zip.ZipFile;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public abstract class AbstractMnistExperiment extends AbstractExperiment {

	private static final int ITERATIONS = 100;
	private static final int SOURCE_SIZE = 200;
	private static final int TARGET_SIZE = 10;

	protected static Instances train = null;
	protected static Instances test = null;

	protected static synchronized void loadData() throws Exception {

		if (train != null)
			return;

		train = getData("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
		train.randomize(new Random(1));
		test = getData("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte");

	}

	private static Instances getData(String labels, String images)
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

	public void runExperiment(SingleSourceTransfer[] classifiers)
			throws Exception {

		AbstractMnistExperiment.loadData();
		Instances[] labeledData = extracedLabledDigits(AbstractMnistExperiment.train);
		Random rand = new Random(1234);

		double[] totals = new double[classifiers.length];
		for (int itr = 0; itr < ITERATIONS; ++itr) {
			Instances source = new Instances(train, 10 * SOURCE_SIZE);
			Instances target = new Instances(train, 10 * TARGET_SIZE);

			for (Instances data : labeledData) {
				data.randomize(rand);
				for (int i = 0; i < SOURCE_SIZE; ++i) {
					Instance instance = data.instance(i);
					source.add(manipulateImage(instance));
				}
				for (int i = 0; i < TARGET_SIZE; ++i) {
					Instance instance = data.instance(i + SOURCE_SIZE);
					target.add(instance);
				}
			}

			for (int i = 0; i < totals.length; ++i) {
				totals[i] += runExperiment(classifiers[i], source, target);
			}
		}
		for (int i = 0; i < totals.length; ++i) {
			System.out.println(ToPerc(totals[i] / ITERATIONS));
		}

	}

	abstract protected Instance manipulateImage(Instance instance);

	private Instances[] extracedLabledDigits(Instances instances) {

		Instances[] labeledSources = new Instances[10];
		for (int i = 0; i < 10; ++i)
			labeledSources[i] = new Instances(instances,
					instances.numInstances());

		@SuppressWarnings("unchecked")
		Enumeration<Instance> enu = instances.enumerateInstances();
		while (enu.hasMoreElements()) {
			Instance instance = enu.nextElement();
			int index = (int) Math.round(instance.classValue());
			labeledSources[index].add(instance);
		}
		for (int i = 0; i < 10; ++i)
			labeledSources[i].compactify();
		return labeledSources;

	}

	private double runExperiment(SingleSourceTransfer classifier,
			Instances source, Instances target) throws Exception {

		source = new Instances(source);
		target = new Instances(target);
		SingleSourceTransfer dup = classifier.makeDuplicate();
		dup.buildModel(source, target);
		return err(dup, test);

	}

}
