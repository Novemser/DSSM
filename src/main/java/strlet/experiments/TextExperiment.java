package strlet.experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import strlet.auxiliary.ThreadPool;
import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public abstract class TextExperiment extends AbstractExperiment {

	private static final int NUM_FEATURES = 51;
	private static final int FOLDS = 20;

	protected static final int DOMAINS = 2;
	protected static File ZIP_FILE;

	protected final String a;
	protected final String b;

	public TextExperiment(String a, String b) {
		this.a = a;
		this.b = b;
	}

	public void runExperiment(SingleSourceTransfer[] classifiers)
			throws Exception {

		LinkedList<String> sourceFiles = getFileNames(0);
		LinkedList<String> targetFiles = getFileNames(1);

		double[] totals = new double[classifiers.length];
		for (int i = 0; i < DOMAINS; ++i) {
			sourceFiles = randomize(sourceFiles);
			targetFiles = randomize(targetFiles);

			for (int fold = 0; fold < FOLDS; ++fold) {
				LinkedList<String> targetTrain = getTrainData(targetFiles, fold);
				LinkedList<String> targetTest = getTestData(targetFiles, fold);

				double[] results = runExperiment(classifiers, sourceFiles,
						targetTrain, targetTest, a, b);
				for (int k = 0; k < totals.length; ++k) {
					totals[k] += results[k];
				}
			}

		}
		for (int i = 0; i < totals.length; ++i) {
			System.out.println(ToPerc(totals[i] / (FOLDS * DOMAINS)));
		}

	}

	private LinkedList<String> getTestData(LinkedList<String> targetFiles,
			int fold) {
		return getData(targetFiles, fold, true);
	}

	private LinkedList<String> getTrainData(LinkedList<String> targetFiles,
			int fold) {
		return getData(targetFiles, fold, false);
	}

	@SuppressWarnings("unused")
	// Making sure 1 part is training data and the rest test data
	private LinkedList<String> getData(LinkedList<String> targetFiles,
			int fold, boolean test) {

		if (FOLDS < 2)
			throw new RuntimeException("Number of folds must be at least 2");
		if (fold >= FOLDS)
			throw new RuntimeException("Fold must be 0 <= fold < FOLDS");

		LinkedList<String> files = new LinkedList<String>();
		int counter = 0;
		for (String f : targetFiles) {
			if (test && counter != fold)
				files.add(f);
			if (!test && counter == fold)
				files.add(f);
			counter = (counter + 1) % FOLDS;
		}
		return files;
	}

	private double[] runExperiment(SingleSourceTransfer[] classifiers,
			LinkedList<String> sourceFiles, LinkedList<String> targetTrain,
			LinkedList<String> targetTest, String classA, String classB)
			throws Exception {

		HashMap<String, Double> idf = getIDF(targetTrain);
		idf = reduceDimentionality(targetTrain, idf, classA, classB);

		HashMap<String, Double> sourceIDF = getIDF(sourceFiles);
		sourceIDF = reduce(sourceIDF, idf.keySet());

		Instances source = createInstances(sourceFiles, sourceIDF, classA,
				classB);
		Instances target = createInstances(targetTrain, idf, classA, classB);
		Instances test = createInstances(targetTest, idf, classA, classB);

		double[] totals = new double[classifiers.length];
		for (int i = 0; i < totals.length; ++i) {
			SingleSourceTransfer dup = classifiers[i].makeDuplicate();
			dup.buildModel(source, target);
			totals[i] += err(dup, test);
		}
		return totals;
	}

	private HashMap<String, Double> reduceDimentionality(
			LinkedList<String> targetTrain, HashMap<String, Double> idf,
			String classA, String classB) throws Exception {

		Instances initial = createInstances(targetTrain, idf, classA, classB);

		AttributeSelection selector = new AttributeSelection();
		selector.setEvaluator(new GainRatioAttributeEval());
		Ranker search = new Ranker();
		search.setNumToSelect(NUM_FEATURES);
		selector.setSearch(search);
		selector.SelectAttributes(initial);
		int[] indexes = selector.selectedAttributes();

		HashSet<String> reducedKeys = new HashSet<String>();
		for (int index : indexes) {
			if (index == initial.classIndex())
				continue;
			String key = initial.attribute(index).name();
			reducedKeys.add(key);
		}

		return reduce(idf, reducedKeys);
	}

	@SuppressWarnings("unchecked")
	private Instances createInstances(LinkedList<String> files,
			HashMap<String, Double> idf, String classA, String classB)
			throws Exception {

		Instances instances = createInstances(idf.keySet(), files.size(),
				classA, classB);
		HashMap<String, Integer> attrIndexes = new HashMap<String, Integer>(
				idf.size());
		for (int i = 0; i < idf.size(); ++i)
			attrIndexes.put(instances.attribute(i).name(), i);

		LinkedList<Callable<Instances>> jobs = new LinkedList<Callable<Instances>>();
		for (int i = 0; i < ThreadPool.poolSize(); ++i)
			jobs.addLast(new FileWorker(instances, files, idf, i, ThreadPool
					.poolSize()));

		LinkedList<Instances> data = new LinkedList<Instances>();
		if (ThreadPool.poolSize() == 1) {
			data.add(jobs.poll().call());
		} else {
			LinkedList<Future<Instances>> futures = new LinkedList<Future<Instances>>();
			while (!jobs.isEmpty())
				futures.add(ThreadPool.submit(jobs.pollFirst()));
			while (!futures.isEmpty())
				data.add(futures.pollFirst().get());
		}

		for (Instances i : data) {
			Enumeration<Instance> enu = i.enumerateInstances();
			while (enu.hasMoreElements())
				instances.add(enu.nextElement());
		}

		return instances;
	}

	private Instances createInstances(Set<String> keySet, int size,
			String classA, String classB) {

		String[] arr = new String[keySet.size()];
		keySet.toArray(arr);
		Arrays.sort(arr);

		FastVector classes = new FastVector(2);
		classes.addElement(classA);
		classes.addElement(classB);

		Attribute[] attrs = new Attribute[arr.length + 1];
		int counter = 0;
		for (String key : arr)
			attrs[counter++] = new Attribute(key);
		attrs[counter] = new Attribute("text_class", classes);

		FastVector attr = new FastVector(attrs.length);
		for (Attribute atribute : attrs)
			attr.addElement(atribute);

		Instances retVal = new Instances("text_data", attr, size);
		retVal.setClassIndex(counter);
		return retVal;
	}

	private HashMap<String, Double> reduce(HashMap<String, Double> idf,
			Set<String> reducedKeys) {

		HashMap<String, Double> retVal = new HashMap<String, Double>();
		for (String key : reducedKeys) {
			double value = 0;
			if (idf.containsKey(key))
				value = idf.get(key);
			retVal.put(key, value);
		}

		return retVal;
	}

	private HashMap<String, Double> getIDF(LinkedList<String> files)
			throws IOException {

		HashMap<String, Integer> counter = new HashMap<String, Integer>();
		ZipFile zip = new ZipFile(ZIP_FILE);
		for (String f : files) {
			ZipEntry entry = zip.getEntry(f);
			HashMap<String, Integer> words = getWords(zip.getInputStream(entry));
			for (String word : words.keySet()) {
				if (!counter.containsKey(word)) {
					counter.put(word, 1);
				} else {
					Integer val = counter.get(word);
					counter.put(word, 1 + val);
				}
			}
		}

		double N = counter.size();
		HashMap<String, Double> idf = new HashMap<String, Double>();
		for (String word : counter.keySet()) {
			double tmp = N / counter.get(word);
			idf.put(word, Math.log(tmp));
		}
		zip.close();
		return idf;
	}

	private HashMap<String, Integer> getWords(InputStream stream)
			throws IOException {
		HashMap<String, Integer> words = new HashMap<String, Integer>();
		BufferedReader br = new BufferedReader(new InputStreamReader(stream));
		String line;
		while ((line = br.readLine()) != null) {
			line = line.trim();
			String[] split = line.split(" ");
			if (notWord(split[0]))
				continue;
			Integer value = Integer.valueOf(split[1]);
			words.put(split[0], value);
		}
		br.close();
		return words;
	}

	private boolean notWord(String word) {

		char[] chars = word.toCharArray();
		for (char c : chars) {
			if (!Character.isLetter(c)) {
				return true;
			}
		}
		return false;

	}

	private LinkedList<String> randomize(LinkedList<String> files) {

		Random rand = new Random(SEED);
		LinkedList<String> tmp = new LinkedList<String>(files);
		LinkedList<String> retVal = new LinkedList<String>();
		while (!tmp.isEmpty()) {
			String f = tmp.remove(rand.nextInt(tmp.size()));
			retVal.add(f);
		}
		return retVal;
	}

	protected abstract LinkedList<String> getFileNames(int fold)
			throws Exception;

	private class FileWorker implements Callable<Instances> {

		private final Instances m_header;
		private final LinkedList<String> m_files;
		private final HashMap<String, Double> m_idf;

		public FileWorker(Instances header, LinkedList<String> files,
				HashMap<String, Double> idf, int start, int step) {
			m_header = new Instances(header, 0);
			m_files = new LinkedList<String>();
			int counter = 0;
			for (String f : files) {
				if (counter++ == start)
					m_files.addLast(f);
				counter %= step;
			}
			m_idf = idf;
		}

		@SuppressWarnings("unchecked")
		@Override
		public Instances call() throws Exception {

			Instances instances = new Instances(m_header, m_files.size());
			String classA = instances.classAttribute().value(0);
			String classB = instances.classAttribute().value(1);

			HashMap<String, Integer> attrIndexes = new HashMap<String, Integer>(
					m_idf.size());
			for (int i = 0; i < m_idf.size(); ++i)
				attrIndexes.put(instances.attribute(i).name(), i);

			ZipFile zip = new ZipFile(ZIP_FILE);
			for (String file : m_files) {
				// System.err.println(file.getName());
				Instance tmp = new Instance(instances.numAttributes());
				tmp.setDataset(instances);

				ZipEntry entry = zip.getEntry(file);
				HashMap<String, Integer> words = getWords(zip
						.getInputStream(entry));
				for (Entry<String, Integer> pair : words.entrySet()) {
					String key = pair.getKey();
					if (!m_idf.containsKey(key))
						continue;
					int f = 1 + words.get(key);
					double tf = Math.log(f);
					double value = tf * m_idf.get(key);

					tmp.setValue(attrIndexes.get(key), value);
				}

				if (isClass(file, classA)) {
					tmp.setClassValue(classA);
				} else {
					tmp.setClassValue(classB);
				}
				instances.add(tmp);
			}
			zip.close();

			Instances retVal = new Instances(instances,
					instances.numInstances());

			Enumeration<Instance> enu = instances.enumerateInstances();
			while (enu.hasMoreElements()) {
				Instance instance = enu.nextElement();
				double[] attValues = instance.toDoubleArray();
				for (int i = 0; i < attValues.length; ++i)
					if (Double.isNaN(attValues[i]))
						attValues[i] = 0;

				Instance newInst = new Instance(instance.weight(), attValues);
				retVal.add(newInst);
			}

			return retVal;
		}
	}

	protected abstract boolean isClass(String name, String classA);

}
