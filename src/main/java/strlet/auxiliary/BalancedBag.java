package strlet.auxiliary;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class BalancedBag {

	private final Instances bag;

	/**
	 * Creates the balanced bag which will be of size
	 * <NUM_CLASSES>*<#MINORITY_CLASS> and without repetitions.
	 * 
	 * @param unbalancedData the Data
	 * @param seed seed for a RNG
	 */
	public BalancedBag(Instances unbalancedData, long seed) {
		this(unbalancedData, seed, 1f);
	}

	/**
	 * Creates the balanced bag which will be of size
	 * <NUM_CLASSES>*<#MINORITY_CLASS>*part and without repetitions.
	 * 
	 * @param unbalancedData the Data
	 * @param seed seed for a RNG
	 * @param part how large a part of the minority class to take
	 */
	public BalancedBag(Instances unbalancedData, long seed, float part) {
		this(unbalancedData, seed, part, false);
	}

	/**
	 * Creates the balanced bag which will be of size
	 * <NUM_CLASSES>*<#MINORITY_CLASS>*part.
	 * 
	 * @param unbalancedData the Data
	 * @param seed seed for a RNG
	 * @param part how large a part of the minority class to take
	 * @param withRepetitions should repetitions be allowed - true to enable
	 */
	@SuppressWarnings("unchecked")
	public BalancedBag(Instances unbalancedData, long seed, float part,
			boolean withRepetitions) {

		int[][] instancesIndx = randomizeInstances(unbalancedData, seed, part,
				withRepetitions);
		int n = 0;
		for (int i = 0; i < instancesIndx.length; ++i)
			n += instancesIndx[i].length;
		bag = new Instances(unbalancedData, n);

		int[] counters = new int[instancesIndx.length];
		int[] indexes = new int[instancesIndx.length];
		Enumeration<Instance> enu = unbalancedData.enumerateInstances();
		while (enu.hasMoreElements()) {
			Instance instance = enu.nextElement();
			int classValue = (int) Math.round(instance.classValue());
			int[] classInstances = instancesIndx[classValue];

			while ((counters[classValue] < classInstances.length)
					&& (classInstances[counters[classValue]] == indexes[classValue])) {
				bag.add(instance);
				++counters[classValue];
			}
			++indexes[classValue];
		}
		bag.compactify();
	}

	private int[][] randomizeInstances(Instances unbalancedData, long seed,
			float part, boolean withRepetitions) {

		int[] sizes = getClassSizes(unbalancedData);
		int minIndex = Utils.minIndex(sizes);
		int minSize = sizes[minIndex];
		while (minSize == 0){
			int maxIndex = Utils.maxIndex(sizes);
			sizes[minIndex] = sizes[maxIndex];
			minIndex = Utils.minIndex(sizes);
			minSize = sizes[minIndex];
		}
		int goalSize = Math.max(Math.round(part * minSize), 1);

		int[][] instances = new int[sizes.length][];
		Random rand = new Random(seed);
		if (withRepetitions) {
			for (int i = 0; i < sizes.length; ++i) {
				instances[i] = new int[goalSize];
				for (int j = 0; j < goalSize; ++j)
					instances[i][j] = rand.nextInt(sizes[i]);
			}
		} else {
			for (int i = 0; i < sizes.length; ++i)
				instances[i] = randomVector(goalSize, sizes[i], rand);
		}

		for (int i = 0; i < sizes.length; ++i)
			Arrays.sort(instances[i]);

		return instances;
	}

	private int[] randomVector(int goalSize, int available, Random rand) {
		int[] origVec = new int[available];
		for (int i = 0; i < available; ++i)
			origVec[i] = i;

		int[] vec = new int[goalSize];
		for (int i = 0; i < goalSize; ++i) {
			int tmp = rand.nextInt(available);
			vec[i] = origVec[tmp];
			origVec[tmp] = origVec[--available];
		}

		return vec;
	}

	@SuppressWarnings("unchecked")
	private int[] getClassSizes(Instances unbalancedData) {

		int[] sizes = new int[unbalancedData.numClasses()];
		Enumeration<Instance> enu = unbalancedData.enumerateInstances();
		while (enu.hasMoreElements()) {
			int classValuse = (int) Math.round(enu.nextElement().classValue());
			++sizes[classValuse];
		}
		return sizes;
	}

	public Instances getBag() {
		return bag;
	}

	public static void main(String[] args) {

		FastVector numericAttribute = getAttributes();
		Instances data1 = new Instances("set1", numericAttribute, 10);
		data1.setClassIndex(1);
		for (int i = 0; i <= 10; ++i) {
			Instance tmp = new Instance(data1.numAttributes());
			tmp.setDataset(data1);
			tmp.setValue(0, i);
			tmp.setValue(1, (i % 2));
			data1.add(tmp);
		}

		System.out.println(data1);
		BalancedBag bag = new BalancedBag(data1, 1l);
		System.out.println(bag.getBag().toString());
		bag = new BalancedBag(data1, 1l, 1f, true);
		System.out.println(bag.getBag().toString());
	}

	public static FastVector getAttributes() {

		FastVector classes = new FastVector(2);
		classes.addElement("a");
		classes.addElement("b");

		Attribute[] attrs = new Attribute[2];
		attrs[0] = new Attribute("attr" + 0);
		attrs[1] = new Attribute("attr" + 1, classes);

		FastVector attr = new FastVector(2);
		for (Attribute atribute : attrs)
			attr.addElement(atribute);

		return attr;
	}

}
