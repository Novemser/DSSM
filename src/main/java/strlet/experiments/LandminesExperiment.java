package strlet.experiments;

import java.io.File;
import java.util.LinkedList;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import strlet.auxiliary.UnderBagging;
import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.supervised.feda.Feda;
import strlet.transferLearning.inductive.taskLearning.naive.SourceOnly;
import strlet.transferLearning.inductive.taskLearning.naive.TargetOnly;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class LandminesExperiment extends AbstractExperiment {

	public void runExperiment(SingleSourceTransfer[] classifiers)
			throws Exception {

		Instances[] fields = new Instances[29];
		ZipFile zip = new ZipFile(new File("testData" + File.separator
				+ "LandmineData.zip"));
		for (int i = 0; i < 29; ++i) {
			ZipEntry entry = zip.getEntry("minefield" + (i + 1) + ".arff");
			DataSource ds = new DataSource(zip.getInputStream(entry));
			fields[i] = ds.getDataSet();
			fields[i].setClassIndex(fields[i].attribute("class").index());
		}
		zip.close();

		LinkedList<Instances> list = new LinkedList<Instances>();
		for (int i = 0; i < 15; ++i) {
			list.add(fields[i]);
		}
		Instances source = combine(list);

		list = new LinkedList<Instances>();
		for (int i = 15; i < 28; ++i) {
			list.add(fields[i]);
		}

		for (SingleSourceTransfer classifier : classifiers) {
			double tot = 0;
			for (int i = 0; i < list.size(); ++i) {
				Instances target = list.removeFirst();
				Instances test = combine(list);
				SingleSourceTransfer dup = classifier.makeDuplicate();
				System.out.println("source, target, test:" + source.numInstances() + "," + target.numInstances() + "," + test.numInstances());
				dup.buildModel(source, target);
				double err = berr(dup, test);
				System.out.println("err:" + err);
				tot += err;
				list.addLast(target);
			}
			System.out.println(ToPerc(tot / list.size()));
		}
	}

	private Instances combine(LinkedList<Instances> list) {
		int total = 0;
		for (Instances set : list) {
			total += set.numInstances();
		}
		Instances combined = new Instances(list.getFirst(), total);
		for (Instances set : list) {
			for (int index = 0; index < set.numInstances(); ++index) {
				combined.add(set.instance(index));
			}
		}
		return combined;
	}

//	public void runExperiment() throws Exception {
//
//		UnderBagging ub = new UnderBagging();
//		ub.setClassifier(new RandomTree());
//		ub.setNumIterations(50);
//		ub.setSeed(SEED);
//
//		Feda feda = new Feda();
//		feda.setBaseClassifier(ub);
//
//		SourceOnly src = new SourceOnly();
//		src.setBaseClassifier(ub);
//
//		TargetOnly tgt = new TargetOnly();
//		tgt.setBaseClassifier(ub);
//
//		SingleSourceTransfer[] models = { src, tgt, feda };
//		// SingleSourceTransfer[] models = { src, tgt };
//		// SingleSourceTransfer[] models = { feda };
//
//		runExperiment(models);
//
//	}

	public static void main(String[] args) throws Exception {
		LandminesExperiment experiment = new LandminesExperiment();
		experiment.runExperiment();
	}

}
