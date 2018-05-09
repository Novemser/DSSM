package strlet.experiments;

import java.io.File;
import java.util.Random;
import java.util.zip.ZipFile;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WineExperiment extends AbstractExperiment {

	public void runExperiment(SingleSourceTransfer[] classifiers)
			throws Exception {

		ZipFile zip = new ZipFile(new File("testData" + File.separator
				+ "wine.zip"));
		DataSource
				ds = new DataSource(zip.getInputStream(zip.getEntry("white.arff")));
		Instances source = ds.getDataSet();
		source.setClassIndex(source.attribute("class").index());

		ds = new DataSource(zip.getInputStream(zip.getEntry("red.arff")));
		Instances target = ds.getDataSet();
		target.setClassIndex(target.attribute("class").index());
		zip.close();

		for (SingleSourceTransfer classifier : classifiers) {
			double tot = 0;
			Instances dup = new Instances(target);
			for (int i = 0; i < 2; ++i) {
				dup.randomize(new Random(SEED * (i + 1)));
				tot += runExperiment(classifier, source, dup);
			}
			System.out.println(ToPerc(tot / 2));
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
		WineExperiment experiment = new WineExperiment();
		experiment.runExperiment();
	}

}
