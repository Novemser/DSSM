package strlet.experiments;

import java.io.File;
import java.util.Random;
import java.util.zip.ZipFile;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DigitsExperiment extends AbstractExperiment {

	public void runExperiment(SingleSourceTransfer[] classifiers)
			throws Exception {

		ZipFile zip = new ZipFile(new File("testData" + File.separator
				+ "digits.zip"));
		DataSource ds = new DataSource(zip.getInputStream(zip
				.getEntry("optdigits_6.arff")));
		Instances six = ds.getDataSet();
		six.setClassIndex(six.attribute("class").index());
		ds = new DataSource(zip.getInputStream(zip.getEntry("optdigits_9.arff")));
		Instances nine = ds.getDataSet();
		nine.setClassIndex(nine.attribute("class").index());
		zip.close();

		for (SingleSourceTransfer classifier : classifiers) {
			double tot = 0;
			Instances typeA = new Instances(six);
			Instances typeB = new Instances(nine);
			for (int i = 0; i < 2; ++i) {
				typeA.randomize(new Random(SEED * (i + 1)));
				typeB.randomize(new Random(SEED * (i + 1)));
				double err1 = runExperiment(classifier, typeA, typeB);
				double err2 = runExperiment(classifier, typeB, typeA);
				tot += err1 + err2;
			}
			System.out.println(ToPerc(tot / 4));
		}
	}

	private double runExperiment(SingleSourceTransfer classifier,
			Instances source, Instances target) throws Exception {

		target = new Instances(target);
		target.stratify(20);
		double total = 0;
		for (int i = 0; i < 20; ++i) {
			Instances train = target.testCV(20, i);
			Instances test = target.trainCV(20, i);
			SingleSourceTransfer dup = classifier.makeDuplicate();
			dup.buildModel(source, train);
			total += err(dup, test);
		}
		return total / 20;

	}

	public static void main(String[] args) throws Exception {
		DigitsExperiment experiment = new DigitsExperiment();
		experiment.runExperiment();
	}

}
