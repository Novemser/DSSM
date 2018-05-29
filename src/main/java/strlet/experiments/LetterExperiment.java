package strlet.experiments;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class LetterExperiment extends AbstractExperiment {

	public void runExperiment(SingleSourceTransfer[] classifiers)
			throws Exception {

		Instances data = loadData();
		Instances[] all = new Instances[data.numClasses()];
		for (int i = 0; i < all.length; ++i) {
			all[i] = new Instances(data, data.numInstances());
		}
		for (int i = 0; i < data.numInstances(); ++i) {
			Instance instance = data.instance(i);
			int classValue = (int) Math.round(instance.classValue());
			all[classValue].add(instance);
		}

		Instances typeA = new Instances(data, data.numInstances());
		Instances typeB = new Instances(data, data.numInstances());
		int attIndex = data.attribute("x2bar").index();
		for (int i = 0; i < all.length; ++i) {
			all[i].compactify();
			double mean = all[i].meanOrMode(attIndex);
			for (int index = 0; index < all[i].numInstances(); ++index) {
				Instance instance = all[i].instance(index);
				if (instance.value(attIndex) < mean) {
					typeA.add(instance);
				} else {
					typeB.add(instance);
				}
			}
		}
		typeA.compactify();
		typeB.compactify();

		for (SingleSourceTransfer classifier : classifiers) {
			Instances dupA = new Instances(typeA);
			Instances dupB = new Instances(typeB);
			double tot = 0;
			for (int i = 0; i < 2; ++i) {
				dupA.randomize(new Random(SEED * (i + 1)));
				dupB.randomize(new Random(SEED * (i + 1)));
				double err1 = runExperiment(classifier, dupA, dupB);
				double err2 = runExperiment(classifier, dupB, dupA);
				System.out.println("Err1, Err2:" + err1 + "," + err2);
				tot += err1;
				tot += err2;
			}
			System.out.println(ToPerc(tot / 4));
		}

	}

	private Instances loadData() throws ZipException, IOException, Exception {
		ZipFile zip = new ZipFile(new File("testData" + File.separator
				+ "letter-recognition.zip"));
		ZipEntry entry = zip.getEntry("letter-recognition.arff");
		DataSource ds = new DataSource(zip.getInputStream(entry));
		Instances data = ds.getDataSet();
		data.setClassIndex(data.attribute("class").index());
		zip.close();
		return data;
	}

	private double runExperiment(SingleSourceTransfer classifier,
			Instances source, Instances target) throws Exception {

//		target = new Instances(target);
//		Instances train = target;
//		Instances test = new Instances(target);
//		SingleSourceTransfer dup = classifier.makeDuplicate();
//		dup.buildModel(source, source);
//		return err(dup, test);
		target = new Instances(target);
		System.out.println("Target total:" + target.numInstances());
		target.stratify(20);
		double total = 0;
		for (int i = 0; i < 20; ++i) {
			Instances train = target.testCV(20, i);
			Instances test = target.trainCV(20, i);
			System.out.println("Train num:" + train.numInstances() + ",. Test num:" + test.numInstances());
			SingleSourceTransfer dup = classifier.makeDuplicate();
			dup.buildModel(source, train);
			total += err(dup, test);
		}
		return total / 20;

	}

	public static void main(String[] args) throws Exception {
		LetterExperiment experiment = new LetterExperiment();
		experiment.runExperiment();
	}

}
