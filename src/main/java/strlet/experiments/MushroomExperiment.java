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
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class MushroomExperiment extends AbstractExperiment {

	public void runExperiment(SingleSourceTransfer[] classifiers)
			throws Exception {

		Instances data = loadData();
		Instances typeA = new Instances(data, data.numInstances());
		Instances typeB = new Instances(data, data.numInstances());
		int attIndex = data.attribute("stalk-shape").index();
		for (int index = 0; index < data.numInstances(); ++index) {
			Instance instance = data.instance(index);
			if (Utils.eq(0, instance.value(attIndex))) {
				typeA.add(instance);
			} else {
				typeB.add(instance);
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
				tot += err1;
				tot += err2;
			}
			System.out.println("Total Err:" + ToPerc(tot / 4));
		}

	}

	private Instances loadData() throws ZipException, IOException, Exception {
		ZipFile zip = new ZipFile(new File("testData" + File.separator
				+ "mushroom.zip"));
		ZipEntry entry = zip.getEntry("mushroom.arff");
		DataSource ds = new DataSource(zip.getInputStream(entry));
		Instances data = ds.getDataSet();
		data.setClassIndex(data.attribute("class").index());
		zip.close();
		return data;
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
		MushroomExperiment experiment = new MushroomExperiment();
		experiment.runExperiment();
	}

}
