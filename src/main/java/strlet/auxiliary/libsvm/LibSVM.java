package strlet.auxiliary.libsvm;

import java.util.Enumeration;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.Vector;

import strlet.auxiliary.libsvm.kernel.Kernel;
import strlet.auxiliary.libsvm.kernel.ONE_CLASS_Q;
import strlet.auxiliary.libsvm.kernel.SVC_Q;
import strlet.auxiliary.libsvm.kernel.SVR_Q;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/*
 * - A wrapper for libsvm, to make it usable in weka
 * - Replaces weka's LibSVM which uses reflections
 */
/**
 * <!-- globalinfo-start --> A wrapper class for the libsvm tools (the libsvm
 * classes, typically the jar file, need to be in the classpath to use this
 * classifier).<br/>
 * LibSVM runs faster than SMO since it uses LibSVM to build the SVM classifier.<br/>
 * LibSVM allows users to experiment with One-class SVM, Regressing SVM, and
 * nu-SVM supported by LibSVM tool. LibSVM reports many useful statistics about
 * LibSVM classifier (e.g., confusion matrix,precision, recall, ROC score,
 * etc.).<br/>
 * <br/>
 * Yasser EL-Manzalawy (2005). WLSVM. URL
 * http://www.cs.iastate.edu/~yasser/wlsvm/.<br/>
 * <br/>
 * Chih-Chung Chang, Chih-Jen Lin (2001). LIBSVM - A Library for Support Vector
 * Machines. URL http://www.csie.ntu.edu.tw/~cjlin/libsvm/.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;misc{EL-Manzalawy2005,
 *    author = {Yasser EL-Manzalawy},
 *    note = {You don't need to include the WLSVM package in the CLASSPATH},
 *    title = {WLSVM},
 *    year = {2005},
 *    URL = {http://www.cs.iastate.edu/\~yasser/wlsvm/}
 * }
 * 
 * &#64;misc{Chang2001,
 *    author = {Chih-Chung Chang and Chih-Jen Lin},
 *    note = {The Weka classifier works with version 2.82 of LIBSVM},
 *    title = {LIBSVM - A Library for Support Vector Machines},
 *    year = {2001},
 *    URL = {http://www.csie.ntu.edu.tw/\~cjlin/libsvm/}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -S &lt;int&gt;
 *  Set type of SVM (default: 0)
 *    0 = C-SVC
 *    1 = nu-SVC
 *    2 = one-class SVM
 *    3 = epsilon-SVR
 *    4 = nu-SVR
 * </pre>
 * 
 * <pre>
 * -K &lt;int&gt;
 *  Set type of kernel function (default: 2)
 *    0 = linear: u'*v
 *    1 = polynomial: (gamma*u'*v + coef0)^degree
 *    2 = radial basis function: exp(-gamma*|u-v|^2)
 *    3 = sigmoid: tanh(gamma*u'*v + coef0)
 * </pre>
 * 
 * <pre>
 * -D &lt;int&gt;
 *  Set degree in kernel function (default: 3)
 * </pre>
 * 
 * <pre>
 * -G &lt;double&gt;
 *  Set gamma in kernel function (default: 1/k)
 * </pre>
 * 
 * <pre>
 * -R &lt;double&gt;
 *  Set coef0 in kernel function (default: 0)
 * </pre>
 * 
 * <pre>
 * -C &lt;double&gt;
 *  Set the parameter C of C-SVC, epsilon-SVR, and nu-SVR
 *   (default: 1)
 * </pre>
 * 
 * <pre>
 * -N &lt;double&gt;
 *  Set the parameter nu of nu-SVC, one-class SVM, and nu-SVR
 *   (default: 0.5)
 * </pre>
 * 
 * <pre>
 * -Z
 *  Turns on normalization of input data (default: off)
 * </pre>
 * 
 * <pre>
 * -J
 *  Turn off nominal to binary conversion.
 *  WARNING: use only if your data is all numeric!
 * </pre>
 * 
 * <pre>
 * -V
 *  Turn off missing value replacement.
 *  WARNING: use only if your data has no missing values.
 * </pre>
 * 
 * <pre>
 * -P &lt;double&gt;
 *  Set the epsilon in loss function of epsilon-SVR (default: 0.1)
 * </pre>
 * 
 * <pre>
 * -M &lt;double&gt;
 *  Set cache memory size in MB (default: 40)
 * </pre>
 * 
 * <pre>
 * -E &lt;double&gt;
 *  Set tolerance of termination criterion (default: 0.001)
 * </pre>
 * 
 * <pre>
 * -H
 *  Turns the shrinking heuristics off (default: on)
 * </pre>
 * 
 * <pre>
 * -W &lt;double&gt;
 *  Set the parameters C of class i to weight[i]*C, for C-SVC
 *  E.g., for a 3-class problem, you could use "1 1 1" for equally
 *  weighted classes.
 *  (default: 1 for all classes)
 * </pre>
 * 
 * <pre>
 * -B
 *  Trains a SVC model instead of a SVR one (default: SVR)
 * </pre>
 * 
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <!-- options-end -->
 */
public class LibSVM extends Classifier {

	private static final long serialVersionUID = 197861816448079817L;

	private static final Random rand = new Random();

	/** LibSVM Model */
	private SVMModel m_Model = null;

	/** normalize input data */
	private boolean m_Normalize = false;

	/** If true, the replace missing values filter is not applied */
	private boolean m_noReplaceMissingValues = false;

	/** SVM types */
	public static final Tag[] TAGS_SVMTYPE = {
			new Tag(SVMType.C_SVC.ordinal(), "C-SVC (classification)"),
			new Tag(SVMType.NU_SVC.ordinal(), "nu-SVC (classification)"),
			new Tag(SVMType.ONE_CLASS_SVM.ordinal(),
					"one-class SVM (classification)"),
			new Tag(SVMType.EPSILON_SVR.ordinal(), "epsilon-SVR (regression)"),
			new Tag(SVMType.NU_SVR.ordinal(), "nu-SVR (regression)") };

	/** the SVM type */
	// private SVMType m_SVMType = SVMType.C_SVC;

	/** the different kernel types */
	public static final Tag[] TAGS_KERNELTYPE = {
			new Tag(KernelType.LINEAR.ordinal(), "linear: u'*v"),
			new Tag(KernelType.POLYNOMIAL.ordinal(),
					"polynomial: (gamma*u'*v + coef0)^degree"),
			new Tag(KernelType.RBF.ordinal(),
					"radial basis function: exp(-gamma*|u-v|^2)"),
			new Tag(KernelType.SIGMOID.ordinal(),
					"sigmoid: tanh(gamma*u'*v + coef0)") };

	/** for poly/rbf/sigmoid */
	private double m_Gamma = 0;

	/** for normalizing the data */
	private Filter m_NormalizeFilter = null;

	/** The filter used to get rid of missing values. */
	private Filter m_ReplaceMissingValues;

	/** All the SVM parameters */
	private final SVMParameter m_parameters;

	public LibSVM() {
		m_parameters = new SVMParameter();
		m_parameters.svm_type = SVMType.C_SVC;
		m_parameters.kernel_type = KernelType.LINEAR;
		m_parameters.degree = 3;
		m_parameters.gamma = 0;
		m_parameters.coef0 = 0;
		m_parameters.nu = 0.5;
		m_parameters.cache_size = 40;
		m_parameters.C = 1;
		m_parameters.eps = 1e-3;
		m_parameters.p = 0.1;
		m_parameters.shrinking = true;
		m_parameters.nr_weight = 0;
		m_parameters.probability = false;
		m_parameters.weight = new double[0];
		m_parameters.weight_label = new int[0];
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option("\tSet type of SVM (default: "
				+ SVMType.C_SVC.ordinal() + ")\n" + "\t\t "
				+ SVMType.C_SVC.ordinal() + " = C-SVC\n" + "\t\t "
				+ SVMType.NU_SVC.ordinal() + " = nu-SVC\n" + "\t\t "
				+ SVMType.ONE_CLASS_SVM.ordinal() + " = one-class SVM\n"
				+ "\t\t " + SVMType.EPSILON_SVR.ordinal() + " = epsilon-SVR\n"
				+ "\t\t " + SVMType.NU_SVR.ordinal() + " = nu-SVR", "S", 1,
				"-S <int>"));

		result.addElement(new Option("\tSet type of kernel function (default: "
				+ KernelType.LINEAR.ordinal() + ")\n" + "\t\t "
				+ KernelType.LINEAR.ordinal() + " = linear: u'*v\n" + "\t\t "
				+ KernelType.POLYNOMIAL.ordinal()
				+ " = polynomial: (gamma*u'*v + coef0)^degree\n" + "\t\t "
				+ KernelType.RBF.ordinal()
				+ " = radial basis function: exp(-gamma*|u-v|^2)\n" + "\t\t "
				+ KernelType.SIGMOID.ordinal()
				+ " = sigmoid: tanh(gamma*u'*v + coef0)", "K", 1, "-K <int>"));

		result.addElement(new Option(
				"\tSet degree in kernel function (default: 3)", "DG", 1,
				"-DG <int>"));

		result.addElement(new Option(
				"\tSet gamma in kernel function (default: 1/k)", "G", 1,
				"-G <double>"));

		result.addElement(new Option(
				"\tSet coef0 in kernel function (default: 0)", "R", 1,
				"-R <double>"));

		result.addElement(new Option(
				"\tSet the parameter C of C-SVC, epsilon-SVR, and nu-SVR\n"
						+ "\t (default: 1)", "C", 1, "-C <double>"));

		result.addElement(new Option(
				"\tSet the parameter nu of nu-SVC, one-class SVM, and nu-SVR\n"
						+ "\t (default: 0.5)", "N", 1, "-N <double>"));

		result.addElement(new Option(
				"\tTurns on normalization of input data (default: off)", "Z",
				0, "-Z"));

		result.addElement(new Option("\tTurn off nominal to binary conversion."
				+ "\n\tWARNING: use only if your data is all numeric!", "J", 0,
				"-J"));

		result.addElement(new Option("\tTurn off missing value replacement."
				+ "\n\tWARNING: use only if your data has no missing "
				+ "values.", "V", 0, "-V"));

		result.addElement(new Option(
				"\tSet the epsilon in loss function of epsilon-SVR (default: 0.1)",
				"P", 1, "-P <double>"));

		result.addElement(new Option(
				"\tSet cache memory size in MB (default: 40)", "M", 1,
				"-M <double>"));

		result.addElement(new Option(
				"\tSet tolerance of termination criterion (default: 0.001)",
				"E", 1, "-E <double>"));

		result.addElement(new Option(
				"\tTurns the shrinking heuristics off (default: on)", "H", 0,
				"-H"));

		result.addElement(new Option(
				"\tSet the parameters C of class i to weight[i]*C, for C-SVC\n"
						+ "\tE.g., for a 3-class problem, you could use \"1 1 1\" for equally\n"
						+ "\tweighted classes.\n"
						+ "\t(default: 1 for all classes)", "W", 1,
				"-W <double>"));

		result.addElement(new Option(
				"\tTrains a SVC model instead of a SVR one (default: SVR)",
				"B", 0, "-B"));

		@SuppressWarnings("unchecked")
		Enumeration<Option> en = super.listOptions();
		while (en.hasMoreElements())
			result.addElement(en.nextElement());

		return result.elements();
	}

	/**
	 * Sets the classifier options
	 * <p/>
	 * 
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 * 
	 * <pre>
	 * -S &lt;int&gt;
	 *  Set type of SVM (default: 0)
	 *    0 = C-SVC
	 *    1 = nu-SVC
	 *    2 = one-class SVM
	 *    3 = epsilon-SVR
	 *    4 = nu-SVR
	 * </pre>
	 * 
	 * <pre>
	 * -K &lt;int&gt;
	 *  Set type of kernel function (default: 0)
	 *    0 = linear: u'*v
	 *    1 = polynomial: (gamma*u'*v + coef0)^degree
	 *    2 = radial basis function: exp(-gamma*|u-v|^2)
	 *    3 = sigmoid: tanh(gamma*u'*v + coef0)
	 * </pre>
	 * 
	 * <pre>
	 * -DG &lt;int&gt;
	 *  Set degree in kernel function (default: 3)
	 * </pre>
	 * 
	 * <pre>
	 * -G &lt;double&gt;
	 *  Set gamma in kernel function (default: 1/k)
	 * </pre>
	 * 
	 * <pre>
	 * -R &lt;double&gt;
	 *  Set coef0 in kernel function (default: 0)
	 * </pre>
	 * 
	 * <pre>
	 * -C &lt;double&gt;
	 *  Set the parameter C of C-SVC, epsilon-SVR, and nu-SVR
	 *   (default: 1)
	 * </pre>
	 * 
	 * <pre>
	 * -N &lt;double&gt;
	 *  Set the parameter nu of nu-SVC, one-class SVM, and nu-SVR
	 *   (default: 0.5)
	 * </pre>
	 * 
	 * <pre>
	 * -Z
	 *  Turns on normalization of input data (default: off)
	 * </pre>
	 * 
	 * <pre>
	 * -J
	 *  Turn off nominal to binary conversion.
	 *  WARNING: use only if your data is all numeric!
	 * </pre>
	 * 
	 * <pre>
	 * -V
	 *  Turn off missing value replacement.
	 *  WARNING: use only if your data has no missing values.
	 * </pre>
	 * 
	 * <pre>
	 * -P &lt;double&gt;
	 *  Set the epsilon in loss function of epsilon-SVR (default: 0.1)
	 * </pre>
	 * 
	 * <pre>
	 * -M &lt;double&gt;
	 *  Set cache memory size in MB (default: 40)
	 * </pre>
	 * 
	 * <pre>
	 * -E &lt;double&gt;
	 *  Set tolerance of termination criterion (default: 0.001)
	 * </pre>
	 * 
	 * <pre>
	 * -H
	 *  Turns the shrinking heuristics off (default: on)
	 * </pre>
	 * 
	 * <pre>
	 * -W &lt;double&gt;
	 *  Set the parameters C of class i to weight[i]*C, for C-SVC
	 *  E.g., for a 3-class problem, you could use "1 1 1" for equally
	 *  weighted classes.
	 *  (default: 1 for all classes)
	 * </pre>
	 * 
	 * <pre>
	 * -B
	 *  Trains a SVC model instead of a SVR one (default: SVR)
	 * </pre>
	 * 
	 * <pre>
	 * -D
	 *  If set, classifier is run in debug mode and
	 *  may output additional info to the console
	 * </pre>
	 * 
	 * <!-- options-end -->
	 * 
	 * @param options
	 *            the options to parse
	 * @throws Exception
	 *             if parsing fails
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		String tmpStr;

		tmpStr = Utils.getOption('S', options);
		if (tmpStr.length() != 0)
			setSVMType(new SelectedTag(Integer.parseInt(tmpStr), TAGS_SVMTYPE));
		else
			setSVMType(new SelectedTag(SVMType.C_SVC.ordinal(), TAGS_SVMTYPE));

		tmpStr = Utils.getOption('K', options);
		if (tmpStr.length() != 0)
			setKernelType(new SelectedTag(Integer.parseInt(tmpStr),
					TAGS_KERNELTYPE));
		else
			setKernelType(new SelectedTag(KernelType.RBF.ordinal(),
					TAGS_KERNELTYPE));

		tmpStr = Utils.getOption("DG", options);
		if (tmpStr.length() != 0)
			setDegree(Integer.parseInt(tmpStr));
		else
			setDegree(3);

		tmpStr = Utils.getOption('G', options);
		if (tmpStr.length() != 0)
			setGamma(Double.parseDouble(tmpStr));
		else
			setGamma(0);

		tmpStr = Utils.getOption('R', options);
		if (tmpStr.length() != 0)
			setCoef0(Double.parseDouble(tmpStr));
		else
			setCoef0(0);

		tmpStr = Utils.getOption('N', options);
		if (tmpStr.length() != 0)
			setNu(Double.parseDouble(tmpStr));
		else
			setNu(0.5);

		tmpStr = Utils.getOption('M', options);
		if (tmpStr.length() != 0)
			setCacheSize(Double.parseDouble(tmpStr));
		else
			setCacheSize(40);

		tmpStr = Utils.getOption('C', options);
		if (tmpStr.length() != 0)
			setCost(Double.parseDouble(tmpStr));
		else
			setCost(1);

		tmpStr = Utils.getOption('E', options);
		if (tmpStr.length() != 0)
			setEps(Double.parseDouble(tmpStr));
		else
			setEps(1e-3);

		setNormalize(Utils.getFlag('Z', options));

		setDoNotReplaceMissingValues(Utils.getFlag("V", options));

		tmpStr = Utils.getOption('P', options);
		if (tmpStr.length() != 0)
			setLoss(Double.parseDouble(tmpStr));
		else
			setLoss(0.1);

		setShrinking(!Utils.getFlag('H', options));

		setWeights(Utils.getOption('W', options));

		setProbabilityEstimates(Utils.getFlag('B', options));

		super.setOptions(options);
	}

	/**
	 * Returns the current options
	 * 
	 * @return the current setup
	 */
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();

		result.add("-S");
		result.add("" + m_parameters.svm_type.ordinal());

		result.add("-K");
		result.add("" + m_parameters.kernel_type.ordinal());

		result.add("-DG");
		result.add("" + getDegree());

		result.add("-G");
		result.add("" + getGamma());

		result.add("-R");
		result.add("" + getCoef0());

		result.add("-N");
		result.add("" + getNu());

		result.add("-M");
		result.add("" + getCacheSize());

		result.add("-C");
		result.add("" + getCost());

		result.add("-E");
		result.add("" + getEps());

		result.add("-P");
		result.add("" + getLoss());

		if (!getShrinking())
			result.add("-H");

		if (getNormalize())
			result.add("-Z");

		if (getDoNotReplaceMissingValues())
			result.add("-V");

		if (getWeights().length() != 0) {
			result.add("-W");
			result.add("" + getWeights());
		}

		if (getProbabilityEstimates())
			result.add("-B");

		String[] superOptions = super.getOptions();
		for (String option : superOptions)
			result.add(option);

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Sets type of SVM (default SVMTYPE_C_SVC)
	 * 
	 * @param value
	 *            the type of the SVM
	 */
	public void setSVMType(SelectedTag value) {
		if (value.getTags() == TAGS_SVMTYPE)
			m_parameters.svm_type = SVMType.values()[value.getSelectedTag()
					.getID()];
	}

	/**
	 * Gets type of SVM
	 * 
	 * @return the type of the SVM
	 */
	public SelectedTag getSVMType() {
		return new SelectedTag(m_parameters.svm_type.ordinal(), TAGS_SVMTYPE);
	}

	/**
	 * Sets type of kernel function (default KERNELTYPE_RBF)
	 * 
	 * @param value
	 *            the kernel type
	 */
	public void setKernelType(SelectedTag value) {
		if (value.getTags() == TAGS_KERNELTYPE)
			m_parameters.kernel_type = KernelType.values()[value
					.getSelectedTag().getID()];
	}

	/**
	 * Gets type of kernel function
	 * 
	 * @return the kernel type
	 */
	public SelectedTag getKernelType() {
		return new SelectedTag(m_parameters.kernel_type.ordinal(),
				TAGS_KERNELTYPE);
	}

	/**
	 * Sets the degree of the kernel
	 * 
	 * @param value
	 *            the degree of the kernel
	 */
	public void setDegree(int value) {
		m_parameters.degree = value;
	}

	/**
	 * Gets the degree of the kernel
	 * 
	 * @return the degree of the kernel
	 */
	public int getDegree() {
		return m_parameters.degree;
	}

	/**
	 * Sets gamma (default = 1/no of attributes)
	 * 
	 * @param value
	 *            the gamma value
	 */
	public void setGamma(double value) {
		m_Gamma = value;
	}

	/**
	 * Gets gamma
	 * 
	 * @return the current gamma
	 */
	public double getGamma() {
		return m_Gamma;
	}

	/**
	 * Sets coef (default 0)
	 * 
	 * @param value
	 *            the coef
	 */
	public void setCoef0(double value) {
		m_parameters.coef0 = value;
	}

	/**
	 * Gets coef
	 * 
	 * @return the coef
	 */
	public double getCoef0() {
		return m_parameters.coef0;
	}

	/**
	 * Sets nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
	 * 
	 * @param value
	 *            the new nu value
	 */
	public void setNu(double value) {
		m_parameters.nu = value;
	}

	/**
	 * Gets nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
	 * 
	 * @return the current nu value
	 */
	public double getNu() {
		return m_parameters.nu;
	}

	/**
	 * Sets cache memory size in MB (default 40)
	 * 
	 * @param value
	 *            the memory size in MB
	 */
	public void setCacheSize(double value) {
		m_parameters.cache_size = value;
	}

	/**
	 * Gets cache memory size in MB
	 * 
	 * @return the memory size in MB
	 */
	public double getCacheSize() {
		return m_parameters.cache_size;
	}

	/**
	 * Sets the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
	 * 
	 * @param value
	 *            the cost value
	 */
	public void setCost(double value) {
		m_parameters.C = value;
	}

	/**
	 * Sets the parameter C of C-SVC, epsilon-SVR, and nu-SVR
	 * 
	 * @return the cost value
	 */
	public double getCost() {
		return m_parameters.C;
	}

	/**
	 * Sets tolerance of termination criterion (default 0.001)
	 * 
	 * @param value
	 *            the tolerance
	 */
	public void setEps(double value) {
		m_parameters.eps = value;
	}

	/**
	 * Gets tolerance of termination criterion
	 * 
	 * @return the current tolerance
	 */
	public double getEps() {
		return m_parameters.eps;
	}

	/**
	 * Sets the epsilon in loss function of epsilon-SVR (default 0.1)
	 * 
	 * @param value
	 *            the loss epsilon
	 */
	public void setLoss(double value) {
		m_parameters.p = value;
	}

	/**
	 * Gets the epsilon in loss function of epsilon-SVR
	 * 
	 * @return the loss epsilon
	 */
	public double getLoss() {
		return m_parameters.p;
	}

	/**
	 * whether to use the shrinking heuristics
	 * 
	 * @param value
	 *            true uses shrinking
	 */
	public void setShrinking(boolean value) {
		m_parameters.shrinking = value;
	}

	/**
	 * whether to use the shrinking heuristics
	 * 
	 * @return true, if shrinking is used
	 */
	public boolean getShrinking() {
		return m_parameters.shrinking;
	}

	/**
	 * whether to normalize input data
	 * 
	 * @param value
	 *            whether to normalize the data
	 */
	public void setNormalize(boolean value) {
		m_Normalize = value;
	}

	/**
	 * whether to normalize input data
	 * 
	 * @return true, if the data is normalized
	 */
	public boolean getNormalize() {
		return m_Normalize;
	}

	/**
	 * Whether to turn off automatic replacement of missing values. Set to true
	 * only if the data does not contain missing values.
	 * 
	 * @param b
	 *            true if automatic missing values replacement is to be
	 *            disabled.
	 */
	public void setDoNotReplaceMissingValues(boolean b) {
		m_noReplaceMissingValues = b;
	}

	/**
	 * Gets whether automatic replacement of missing values is disabled.
	 * 
	 * @return true if automatic replacement of missing values is disabled.
	 */
	public boolean getDoNotReplaceMissingValues() {
		return m_noReplaceMissingValues;
	}

	/**
	 * Sets the parameters C of class i to weight[i]*C, for C-SVC (default 1).
	 * Blank separated list of doubles.
	 * 
	 * @param weightsStr
	 *            the weights (doubles, separated by blanks)
	 */
	public void setWeights(String weightsStr) {
		StringTokenizer tok;
		int i;

		tok = new StringTokenizer(weightsStr, " ");
		m_parameters.weight = new double[tok.countTokens()];
		m_parameters.nr_weight = m_parameters.weight.length;
		m_parameters.weight_label = new int[tok.countTokens()];

		if (m_parameters.nr_weight == 0)
			System.out
					.println("Zero Weights processed. Default weights will be used");

		for (i = 0; i < m_parameters.nr_weight; i++) {
			m_parameters.weight[i] = Double.parseDouble(tok.nextToken());
			m_parameters.weight_label[i] = i;
		}
	}

	/**
	 * Gets the parameters C of class i to weight[i]*C, for C-SVC (default 1).
	 * Blank separated doubles.
	 * 
	 * @return the weights (doubles separated by blanks)
	 */
	public String getWeights() {
		String result;
		int i;

		result = "";
		for (i = 0; i < m_parameters.weight.length; i++) {
			if (i > 0)
				result += " ";
			result += Double.toString(m_parameters.weight[i]);
		}

		return result;
	}

	/**
	 * Returns whether probability estimates are generated instead of -1/+1 for
	 * classification problems.
	 * 
	 * @param value
	 *            whether to predict probabilities
	 */
	public void setProbabilityEstimates(boolean value) {
		m_parameters.probability = value;
	}

	/**
	 * Sets whether to generate probability estimates instead of -1/+1 for
	 * classification problems.
	 * 
	 * @return true, if probability estimates should be returned
	 */
	public boolean getProbabilityEstimates() {
		return m_parameters.probability;
	}

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enableDependency(Capability.UNARY_CLASS);
		result.enableDependency(Capability.NOMINAL_CLASS);
		result.enableDependency(Capability.NUMERIC_CLASS);
		result.enableDependency(Capability.DATE_CLASS);

		switch (m_parameters.svm_type) {
		case C_SVC:
		case NU_SVC:
			result.enable(Capability.NOMINAL_CLASS);
			break;

		case ONE_CLASS_SVM:
			result.enable(Capability.UNARY_CLASS);
			break;

		case EPSILON_SVR:
		case NU_SVR:
			result.enable(Capability.NUMERIC_CLASS);
			result.enable(Capability.DATE_CLASS);
			break;

		default:
			throw new IllegalArgumentException("SVMType "
					+ m_parameters.svm_type + " is not supported!");
		}
		result.enable(Capability.MISSING_CLASS_VALUES);

		return result;
	}

	/**
	 * builds the classifier
	 * 
	 * @param insts
	 *            the training instances
	 * @throws Exception
	 *             if libsvm classes not in classpath or libsvm encountered a
	 *             problem
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception {

		Instances insts = preProcessData(data);

		SVMNode[][] instancesData = new SVMNode[insts.numInstances()][];
		double[] classifications = new double[insts.numInstances()];
		boolean[] used = new boolean[insts.numAttributes()];
		for (int d = 0; d < instancesData.length; ++d) {
			Instance inst = insts.instance(d);
			SVMNode[] instanceData = instanceToArray(inst);
			instancesData[d] = instanceData;
			classifications[d] = inst.classValue();
			for (SVMNode node : instanceData)
				used[node.index] = true;
		}

		if (getGamma() == 0) {
			int counter = 0;
			for (boolean b : used)
				if (b)
					++counter;
			m_parameters.gamma = 1.0 / counter;
		} else
			m_parameters.gamma = m_Gamma;

		// train model
		SVMProblem prob = getProblem(instancesData, classifications);
		String error_msg = checkParameter(prob, m_parameters);
		if (error_msg != null)
			throw new Exception("Error: " + error_msg);
		m_Model = train(prob, m_parameters);
	}

	private SVMModel train(SVMProblem prob, SVMParameter param) {

		if (param.svm_type.equals(SVMType.ONE_CLASS_SVM)
				|| param.svm_type.equals(SVMType.EPSILON_SVR)
				|| param.svm_type.equals(SVMType.NU_SVR)) // regression or
															// one-class-svm
			return trainOneClass(prob, param);
		else { // classification
			return trainClassifier(prob, param);
		}
	}

	private SVMModel trainClassifier(SVMProblem prob, SVMParameter param) {

		SVMModel model = new SVMModel();
		model.param = param;

		int l = prob.l;
		int[] tmp_nr_class = new int[1];
		int[][] tmp_label = new int[1][];
		int[][] tmp_start = new int[1][];
		int[][] tmp_count = new int[1][];
		int[] perm = new int[l];

		// group training data of the same class
		groupClasses(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count, perm);
		int nr_class = tmp_nr_class[0];
		int[] label = tmp_label[0];
		int[] start = tmp_start[0];
		int[] count = tmp_count[0];

		SVMNode[][] x = new SVMNode[l][];
		int i;
		for (i = 0; i < l; i++)
			x[i] = prob.x[perm[i]];

		// calculate weighted C

		double[] weighted_C = new double[nr_class];
		for (i = 0; i < nr_class; i++)
			weighted_C[i] = param.C;
		for (i = 0; i < param.nr_weight; i++) {
			int j;
			for (j = 0; j < nr_class; j++)
				if (param.weight_label[i] == label[j])
					break;
			if (j != nr_class)
				weighted_C[j] *= param.weight[i];
		}

		// train k*(k-1)/2 models

		boolean[] nonzero = new boolean[l];
		for (i = 0; i < l; i++)
			nonzero[i] = false;
		DecisionFunction[] decisionFunctions = new DecisionFunction[nr_class * (nr_class - 1)
				/ 2];

		double[] probA = null, probB = null;
		if (param.probability) {
			probA = new double[nr_class * (nr_class - 1) / 2];
			probB = new double[nr_class * (nr_class - 1) / 2];
		}

		int p = 0;
		for (i = 0; i < nr_class; i++)
			for (int j = i + 1; j < nr_class; j++) {
				SVMProblem sub_prob = new SVMProblem();
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.l = ci + cj;
				sub_prob.x = new SVMNode[sub_prob.l][];
				sub_prob.y = new double[sub_prob.l];
				int k;
				for (k = 0; k < ci; k++) {
					sub_prob.x[k] = x[si + k];
					sub_prob.y[k] = +1;
				}
				for (k = 0; k < cj; k++) {
					sub_prob.x[ci + k] = x[sj + k];
					sub_prob.y[ci + k] = -1;
				}

				if (param.probability) {
					double[] probAB = new double[2];
					binarySVCProbability(sub_prob, param, weighted_C[i],
							weighted_C[j], probAB);
					probA[p] = probAB[0];
					probB[p] = probAB[1];
				}

				decisionFunctions[p] = trainOne(sub_prob, param, weighted_C[i], weighted_C[j]);
				for (k = 0; k < ci; k++)
					if (!nonzero[si + k] && Math.abs(decisionFunctions[p].alpha[k]) > 0)
						nonzero[si + k] = true;
				for (k = 0; k < cj; k++)
					if (!nonzero[sj + k] && Math.abs(decisionFunctions[p].alpha[ci + k]) > 0)
						nonzero[sj + k] = true;
				++p;
			}

		// build output

		model.nr_class = nr_class;

		model.label = new int[nr_class];
		for (i = 0; i < nr_class; i++)
			model.label[i] = label[i];

		model.rho = new double[nr_class * (nr_class - 1) / 2];
		for (i = 0; i < nr_class * (nr_class - 1) / 2; i++)
			model.rho[i] = decisionFunctions[i].rho;

		if (param.probability) {
			model.probA = new double[nr_class * (nr_class - 1) / 2];
			model.probB = new double[nr_class * (nr_class - 1) / 2];
			for (i = 0; i < nr_class * (nr_class - 1) / 2; i++) {
				model.probA[i] = probA[i];
				model.probB[i] = probB[i];
			}
		} else {
			model.probA = null;
			model.probB = null;
		}

		int nnz = 0;
		int[] nz_count = new int[nr_class];
		model.nSV = new int[nr_class];
		for (i = 0; i < nr_class; i++) {
			int nSV = 0;
			for (int j = 0; j < count[i]; j++)
				if (nonzero[start[i] + j]) {
					++nSV;
					++nnz;
				}
			model.nSV[i] = nSV;
			nz_count[i] = nSV;
		}

		model.l = nnz;
		model.SV = new SVMNode[nnz][];
		model.sv_indices = new int[nnz];
		p = 0;
		for (i = 0; i < l; i++)
			if (nonzero[i]) {
				model.SV[p] = x[i];
				model.sv_indices[p++] = perm[i] + 1;
			}

		int[] nz_start = new int[nr_class];
		nz_start[0] = 0;
		for (i = 1; i < nr_class; i++)
			nz_start[i] = nz_start[i - 1] + nz_count[i - 1];

		model.sv_coef = new double[nr_class - 1][];
		for (i = 0; i < nr_class - 1; i++)
			model.sv_coef[i] = new double[nnz];

		p = 0;
		for (i = 0; i < nr_class; i++)
			for (int j = i + 1; j < nr_class; j++) {
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];

				int q = nz_start[i];
				int k;
				for (k = 0; k < ci; k++)
					if (nonzero[si + k])
						model.sv_coef[j - 1][q++] = decisionFunctions[p].alpha[k];
				q = nz_start[j];
				for (k = 0; k < cj; k++)
					if (nonzero[sj + k])
						model.sv_coef[i][q++] = decisionFunctions[p].alpha[ci + k];
				++p;
			}
		return model;
	}

	// Cross-validation decision values for probability estimates
	private void binarySVCProbability(SVMProblem prob, SVMParameter param,
			double Cp, double Cn, double[] probAB) {
		int i;
		int nr_fold = 5;
		int[] perm = new int[prob.l];
		double[] dec_values = new double[prob.l];

		// random shuffle
		for (i = 0; i < prob.l; i++)
			perm[i] = i;
		for (i = 0; i < prob.l; i++) {
			int j = i + rand.nextInt(prob.l - i);
			do {
				int _x = perm[i];
				perm[i] = perm[j];
				perm[j] = _x;
			} while (false);
		}
		for (i = 0; i < nr_fold; i++) {
			int begin = i * prob.l / nr_fold;
			int end = (i + 1) * prob.l / nr_fold;
			int j, k;
			SVMProblem subprob = new SVMProblem();

			subprob.l = prob.l - (end - begin);
			subprob.x = new SVMNode[subprob.l][];
			subprob.y = new double[subprob.l];

			k = 0;
			for (j = 0; j < begin; j++) {
				subprob.x[k] = prob.x[perm[j]];
				subprob.y[k] = prob.y[perm[j]];
				++k;
			}
			for (j = end; j < prob.l; j++) {
				subprob.x[k] = prob.x[perm[j]];
				subprob.y[k] = prob.y[perm[j]];
				++k;
			}
			int p_count = 0, n_count = 0;
			for (j = 0; j < k; j++)
				if (subprob.y[j] > 0)
					p_count++;
				else
					n_count++;

			if (p_count == 0 && n_count == 0)
				for (j = begin; j < end; j++)
					dec_values[perm[j]] = 0;
			else if (p_count > 0 && n_count == 0)
				for (j = begin; j < end; j++)
					dec_values[perm[j]] = 1;
			else if (p_count == 0 && n_count > 0)
				for (j = begin; j < end; j++)
					dec_values[perm[j]] = -1;
			else {
				SVMParameter subparam = (SVMParameter) param.clone();
				subparam.probability = false;
				subparam.C = 1.0;
				subparam.nr_weight = 2;
				subparam.weight_label = new int[2];
				subparam.weight = new double[2];
				subparam.weight_label[0] = +1;
				subparam.weight_label[1] = -1;
				subparam.weight[0] = Cp;
				subparam.weight[1] = Cn;
				SVMModel submodel = train(subprob, subparam);
				for (j = begin; j < end; j++) {
					double[] dec_value = new double[1];
					predictValues(submodel, prob.x[perm[j]], dec_value);
					dec_values[perm[j]] = dec_value[0];
					// ensure +1 -1 order; reason not using CV subroutine
					dec_values[perm[j]] *= submodel.label[0];
				}
			}
		}
		sigmoidTrain(prob.l, dec_values, prob.y, probAB);
	}

	// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
	private void sigmoidTrain(int l, double[] dec_values, double[] labels,
			double[] probAB) {
		double A, B;
		double prior1 = 0, prior0 = 0;
		int i;

		for (i = 0; i < l; i++)
			if (labels[i] > 0)
				prior1 += 1;
			else
				prior0 += 1;

		int max_iter = 100; // Maximal number of iterations
		double min_step = 1e-10; // Minimal step taken in line search
		double sigma = 1e-12; // For numerically strict PD of Hessian
		double eps = 1e-5;
		double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
		double loTarget = 1 / (prior0 + 2.0);
		double[] t = new double[l];
		double fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize;
		double newA, newB, newf, d1, d2;
		int iter;

		// Initial Point and Initial Fun Value
		A = 0.0;
		B = Math.log((prior0 + 1.0) / (prior1 + 1.0));
		double fval = 0.0;

		for (i = 0; i < l; i++) {
			if (labels[i] > 0)
				t[i] = hiTarget;
			else
				t[i] = loTarget;
			fApB = dec_values[i] * A + B;
			if (fApB >= 0)
				fval += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
			else
				fval += (t[i] - 1) * fApB + Math.log(1 + Math.exp(fApB));
		}
		for (iter = 0; iter < max_iter; iter++) {
			// Update Gradient and Hessian (use H' = H + sigma I)
			h11 = sigma; // numerically ensures strict PD
			h22 = sigma;
			h21 = 0.0;
			g1 = 0.0;
			g2 = 0.0;
			for (i = 0; i < l; i++) {
				fApB = dec_values[i] * A + B;
				if (fApB >= 0) {
					p = Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
					q = 1.0 / (1.0 + Math.exp(-fApB));
				} else {
					p = 1.0 / (1.0 + Math.exp(fApB));
					q = Math.exp(fApB) / (1.0 + Math.exp(fApB));
				}
				d2 = p * q;
				h11 += dec_values[i] * dec_values[i] * d2;
				h22 += d2;
				h21 += dec_values[i] * d2;
				d1 = t[i] - p;
				g1 += dec_values[i] * d1;
				g2 += d1;
			}

			// Stopping Criteria
			if (Math.abs(g1) < eps && Math.abs(g2) < eps)
				break;

			// Finding Newton direction: -inv(H') * g
			det = h11 * h22 - h21 * h21;
			dA = -(h22 * g1 - h21 * g2) / det;
			dB = -(-h21 * g1 + h11 * g2) / det;
			gd = g1 * dA + g2 * dB;

			stepsize = 1; // Line Search
			while (stepsize >= min_step) {
				newA = A + stepsize * dA;
				newB = B + stepsize * dB;

				// New function value
				newf = 0.0;
				for (i = 0; i < l; i++) {
					fApB = dec_values[i] * newA + newB;
					if (fApB >= 0)
						newf += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
					else
						newf += (t[i] - 1) * fApB
								+ Math.log(1 + Math.exp(fApB));
				}
				// Check sufficient decrease
				if (newf < fval + 0.0001 * stepsize * gd) {
					A = newA;
					B = newB;
					fval = newf;
					break;
				} else
					stepsize = stepsize / 2.0;
			}

			if (stepsize < min_step)
				break;
		}

		probAB[0] = A;
		probAB[1] = B;
	}

	private SVMModel trainOneClass(SVMProblem prob, SVMParameter param) {

		SVMModel model = new SVMModel();
		model.param = param;

		model.nr_class = 2;
		model.label = null;
		model.nSV = null;
		model.probA = null;
		model.probB = null;
		model.sv_coef = new double[1][];

		if (param.probability
				&& (param.svm_type.equals(SVMType.EPSILON_SVR) || param.svm_type
						.equals(SVMType.NU_SVC))) {
			model.probA = new double[1];
			model.probA[0] = calcSvrProbability(prob, param);
		}

		DecisionFunction f = trainOne(prob, param, 0, 0);
		model.rho = new double[1];
		model.rho[0] = f.rho;

		int nSV = 0;
		for (int i = 0; i < prob.l; ++i)
			if (Math.abs(f.alpha[i]) > 0)
				++nSV;
		model.l = nSV;
		model.SV = new SVMNode[nSV][];
		model.sv_coef[0] = new double[nSV];
		model.sv_indices = new int[nSV];
		int j = 0;
		for (int i = 0; i < prob.l; ++i)
			if (Math.abs(f.alpha[i]) > 0) {
				model.SV[j] = prob.x[i];
				model.sv_coef[0][j] = f.alpha[i];
				model.sv_indices[j] = i + 1;
				++j;
			}
		return model;
	}

	private DecisionFunction trainOne(SVMProblem prob, SVMParameter param,
			double Cp, double Cn) {
		double[] alpha = new double[prob.l];
		SolutionInfo si = new SolutionInfo();
		switch (param.svm_type) {
		case C_SVC:
			solveCSVC(prob, param, alpha, si, Cp, Cn);
			break;
		case NU_SVC:
			solveNUSVC(prob, param, alpha, si);
			break;
		case ONE_CLASS_SVM:
			solveOneClass(prob, param, alpha, si);
			break;
		case EPSILON_SVR:
			solveEpsilonSVR(prob, param, alpha, si);
			break;
		case NU_SVR:
			solveNUSVR(prob, param, alpha, si);
			break;
		}

		DecisionFunction f = new DecisionFunction();
		f.alpha = alpha;
		f.rho = si.rho;
		return f;
	}

	private void solveCSVC(SVMProblem prob, SVMParameter param, double[] alpha,
			SolutionInfo si, double Cp, double Cn) {
		int l = prob.l;
		double[] minus_ones = new double[l];
		byte[] y = new byte[l];

		int i;

		for (i = 0; i < l; i++) {
			alpha[i] = 0;
			minus_ones[i] = -1;
			if (prob.y[i] > 0)
				y[i] = +1;
			else
				y[i] = -1;
		}

		Solver s = new Solver();
		s.Solve(l, new SVC_Q(prob, param, y), minus_ones, y, alpha, Cp, Cn,
				param.eps, si, param.shrinking);

		for (i = 0; i < l; i++)
			alpha[i] *= y[i];
	}

	private void solveNUSVC(SVMProblem prob, SVMParameter param,
			double[] alpha, SolutionInfo si) {
		int i;
		int l = prob.l;
		double nu = param.nu;

		byte[] y = new byte[l];

		for (i = 0; i < l; i++)
			if (prob.y[i] > 0)
				y[i] = +1;
			else
				y[i] = -1;

		double sum_pos = nu * l / 2;
		double sum_neg = nu * l / 2;

		for (i = 0; i < l; i++)
			if (y[i] == +1) {
				alpha[i] = Math.min(1.0, sum_pos);
				sum_pos -= alpha[i];
			} else {
				alpha[i] = Math.min(1.0, sum_neg);
				sum_neg -= alpha[i];
			}

		double[] zeros = new double[l];

		for (i = 0; i < l; i++)
			zeros[i] = 0;

		Solver_NU s = new Solver_NU();
		s.Solve(l, new SVC_Q(prob, param, y), zeros, y, alpha, 1.0, 1.0,
				param.eps, si, param.shrinking);
		double r = si.r;

		for (i = 0; i < l; i++)
			alpha[i] *= y[i] / r;

		si.rho /= r;
	}

	private void solveOneClass(SVMProblem prob, SVMParameter param,
			double[] alpha, SolutionInfo si) {
		int l = prob.l;
		double[] zeros = new double[l];
		byte[] ones = new byte[l];
		int i;

		int n = (int) (param.nu * prob.l); // # of alpha's at upper bound

		for (i = 0; i < n; i++)
			alpha[i] = 1;
		if (n < prob.l)
			alpha[n] = param.nu * prob.l - n;
		for (i = n + 1; i < l; i++)
			alpha[i] = 0;

		for (i = 0; i < l; i++) {
			zeros[i] = 0;
			ones[i] = 1;
		}

		Solver s = new Solver();
		s.Solve(l, new ONE_CLASS_Q(prob, param), zeros, ones, alpha, 1.0, 1.0,
				param.eps, si, param.shrinking);
	}

	private void solveEpsilonSVR(SVMProblem prob, SVMParameter param,
			double[] alpha, SolutionInfo si) {

		int l = prob.l;
		double[] alpha2 = new double[2 * l];
		double[] linear_term = new double[2 * l];
		byte[] y = new byte[2 * l];
		int i;

		for (i = 0; i < l; i++) {
			alpha2[i] = 0;
			linear_term[i] = param.p - prob.y[i];
			y[i] = 1;

			alpha2[i + l] = 0;
			linear_term[i + l] = param.p + prob.y[i];
			y[i + l] = -1;
		}

		Solver s = new Solver();
		s.Solve(2 * l, new SVR_Q(prob, param), linear_term, y, alpha2, param.C,
				param.C, param.eps, si, param.shrinking);
	}

	private void solveNUSVR(SVMProblem prob, SVMParameter param,
			double[] alpha, SolutionInfo si) {
		int l = prob.l;
		double C = param.C;
		double[] alpha2 = new double[2 * l];
		double[] linear_term = new double[2 * l];
		byte[] y = new byte[2 * l];
		int i;

		double sum = C * param.nu * l / 2;
		for (i = 0; i < l; i++) {
			alpha2[i] = alpha2[i + l] = Math.min(sum, C);
			sum -= alpha2[i];

			linear_term[i] = -prob.y[i];
			y[i] = 1;

			linear_term[i + l] = prob.y[i];
			y[i + l] = -1;
		}

		Solver_NU s = new Solver_NU();
		s.Solve(2 * l, new SVR_Q(prob, param), linear_term, y, alpha2, C, C,
				param.eps, si, param.shrinking);

		for (i = 0; i < l; i++)
			alpha[i] = alpha2[i] - alpha2[i + l];
	}

	private double calcSvrProbability(SVMProblem prob, SVMParameter param) {

		int nr_fold = 5;
		double[] ymv = new double[prob.l];

		SVMParameter newparam = (SVMParameter) param.clone();
		newparam.probability = false;
		crossValidation(prob, newparam, nr_fold, ymv);
		double mae = 0;
		for (int i = 0; i < prob.l; ++i) {
			ymv[i] = prob.y[i] - ymv[i];
			mae += Math.abs(ymv[i]);
		}
		mae /= prob.l;
		double std = Math.sqrt(2 * mae * mae);
		int count = 0;
		mae = 0;
		for (int i = 0; i < prob.l; ++i)
			if (Math.abs(ymv[i]) > 5 * std)
				count = count + 1;
			else
				mae += Math.abs(ymv[i]);
		mae /= (prob.l - count);
		return mae;
	}

	private void crossValidation(SVMProblem prob, SVMParameter param,
			int nr_fold, double[] target) {

		int i;
		int[] fold_start = new int[nr_fold + 1];
		int l = prob.l;
		int[] perm = new int[l];

		// stratified cv may not give leave-one-out rate
		// Each class to l folds -> some folds may have zero elements
		if ((param.svm_type.equals(SVMType.C_SVC) || param.svm_type
				.equals(SVMType.NU_SVC)) && nr_fold < l) {
			int[] tmp_nr_class = new int[1];
			int[][] tmp_label = new int[1][];
			int[][] tmp_start = new int[1][];
			int[][] tmp_count = new int[1][];

			groupClasses(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count,
					perm);

			int nr_class = tmp_nr_class[0];
			int[] start = tmp_start[0];
			int[] count = tmp_count[0];

			// random shuffle and then data grouped by fold using the array perm
			int[] fold_count = new int[nr_fold];
			int c;
			int[] index = new int[l];
			for (i = 0; i < l; i++)
				index[i] = perm[i];
			for (c = 0; c < nr_class; c++)
				for (i = 0; i < count[c]; i++) {
					int j = i + rand.nextInt(count[c] - i);
					do {
						int _x = index[start[c] + j];
						index[start[c] + j] = index[start[c] + i];
						index[start[c] + i] = _x;
					} while (false);
				}
			for (i = 0; i < nr_fold; i++) {
				fold_count[i] = 0;
				for (c = 0; c < nr_class; c++)
					fold_count[i] += (i + 1) * count[c] / nr_fold - i
							* count[c] / nr_fold;
			}
			fold_start[0] = 0;
			for (i = 1; i <= nr_fold; i++)
				fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
			for (c = 0; c < nr_class; c++)
				for (i = 0; i < nr_fold; i++) {
					int begin = start[c] + i * count[c] / nr_fold;
					int end = start[c] + (i + 1) * count[c] / nr_fold;
					for (int j = begin; j < end; j++) {
						perm[fold_start[i]] = index[j];
						fold_start[i]++;
					}
				}
			fold_start[0] = 0;
			for (i = 1; i <= nr_fold; i++)
				fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
		} else {
			for (i = 0; i < l; i++)
				perm[i] = i;
			for (i = 0; i < l; i++) {
				int j = i + rand.nextInt(l - i);
				do {
					int _x = perm[i];
					perm[i] = perm[j];
					perm[j] = _x;
				} while (false);
			}
			for (i = 0; i <= nr_fold; i++)
				fold_start[i] = i * l / nr_fold;
		}

		for (i = 0; i < nr_fold; i++) {
			int begin = fold_start[i];
			int end = fold_start[i + 1];
			int j, k;
			SVMProblem subprob = new SVMProblem();

			subprob.l = l - (end - begin);
			subprob.x = new SVMNode[subprob.l][];
			subprob.y = new double[subprob.l];

			k = 0;
			for (j = 0; j < begin; j++) {
				subprob.x[k] = prob.x[perm[j]];
				subprob.y[k] = prob.y[perm[j]];
				++k;
			}
			for (j = end; j < l; j++) {
				subprob.x[k] = prob.x[perm[j]];
				subprob.y[k] = prob.y[perm[j]];
				++k;
			}
			SVMModel submodel = train(subprob, param);
			if (param.probability
					&& (param.svm_type.equals(SVMType.C_SVC) || param.svm_type
							.equals(SVMType.NU_SVC))) {
				double[] prob_estimates = new double[submodel.nr_class];
				for (j = begin; j < end; j++)
					target[perm[j]] = predictProbability(submodel,
							prob.x[perm[j]], prob_estimates);
			} else
				for (j = begin; j < end; j++)
					target[perm[j]] = predict(submodel, prob.x[perm[j]]);
		}
	}

	private double predict(SVMModel model, SVMNode[] x) {
		int nr_class = model.nr_class;
		double[] dec_values;
		if (model.param.svm_type.equals(SVMType.ONE_CLASS_SVM)
				|| model.param.svm_type.equals(SVMType.EPSILON_SVR)
				|| model.param.svm_type.equals(SVMType.NU_SVR))
			dec_values = new double[1];
		else
			dec_values = new double[nr_class * (nr_class - 1) / 2];
		double pred_result = predictValues(model, x, dec_values);
		return pred_result;
	}

	private double predictProbability(SVMModel model, SVMNode[] x,
			double[] prob_estimates) {

		if ((model.param.svm_type.equals(SVMType.C_SVC) || model.param.svm_type
				.equals(SVMType.NU_SVC))
				&& model.probA != null
				&& model.probB != null) {
			int i;
			int nr_class = model.nr_class;
			double[] dec_values = new double[nr_class * (nr_class - 1) / 2];
			predictValues(model, x, dec_values);

			double min_prob = 1e-7;
			double[][] pairwise_prob = new double[nr_class][nr_class];

			int k = 0;
			for (i = 0; i < nr_class; i++)
				for (int j = i + 1; j < nr_class; j++) {
					pairwise_prob[i][j] = Math.min(Math.max(
							sigmoidPredict(dec_values[k], model.probA[k],
									model.probB[k]), min_prob), 1 - min_prob);
					pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
					k++;
				}
			multiclassProbability(nr_class, pairwise_prob, prob_estimates);

			int prob_max_idx = 0;
			for (i = 1; i < nr_class; i++)
				if (prob_estimates[i] > prob_estimates[prob_max_idx])
					prob_max_idx = i;
			return model.label[prob_max_idx];
		} else
			return predict(model, x);
	}

	// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
	private void multiclassProbability(int k, double[][] r, double[] p) {
		int t, j;
		int iter = 0, max_iter = Math.max(100, k);
		double[][] Q = new double[k][k];
		double[] Qp = new double[k];
		double pQp, eps = 0.005 / k;

		for (t = 0; t < k; t++) {
			p[t] = 1.0 / k; // Valid if k = 1
			Q[t][t] = 0;
			for (j = 0; j < t; j++) {
				Q[t][t] += r[j][t] * r[j][t];
				Q[t][j] = Q[j][t];
			}
			for (j = t + 1; j < k; j++) {
				Q[t][t] += r[j][t] * r[j][t];
				Q[t][j] = -r[j][t] * r[t][j];
			}
		}
		for (iter = 0; iter < max_iter; iter++) {
			// stopping condition, recalculate QP,pQP for numerical accuracy
			pQp = 0;
			for (t = 0; t < k; t++) {
				Qp[t] = 0;
				for (j = 0; j < k; j++)
					Qp[t] += Q[t][j] * p[j];
				pQp += p[t] * Qp[t];
			}
			double max_error = 0;
			for (t = 0; t < k; t++) {
				double error = Math.abs(Qp[t] - pQp);
				if (error > max_error)
					max_error = error;
			}
			if (max_error < eps)
				break;

			for (t = 0; t < k; t++) {
				double diff = (-Qp[t] + pQp) / Q[t][t];
				p[t] += diff;
				pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff)
						/ (1 + diff);
				for (j = 0; j < k; j++) {
					Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
					p[j] /= (1 + diff);
				}
			}
		}
	}

	private static double sigmoidPredict(double decision_value, double A,
			double B) {
		double fApB = decision_value * A + B;
		if (fApB >= 0)
			return Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
		else
			return 1.0 / (1 + Math.exp(fApB));
	}

	private double predictValues(SVMModel model, SVMNode[] x,
			double[] dec_values) {
		int i;
		if (model.param.svm_type.equals(SVMType.ONE_CLASS_SVM)
				|| model.param.svm_type.equals(SVMType.EPSILON_SVR)
				|| model.param.svm_type.equals(SVMType.NU_SVR)) {
			double[] sv_coef = model.sv_coef[0];
			double sum = 0;
			for (i = 0; i < model.l; i++)
				sum += sv_coef[i]
						* Kernel.k_function(x, model.SV[i], model.param);
			sum -= model.rho[0];
			dec_values[0] = sum;

			if (model.param.svm_type.equals(SVMType.ONE_CLASS_SVM))
				return (sum > 0) ? 1 : -1;
			else
				return sum;
		} else {
			int nr_class = model.nr_class;
			int l = model.l;

			double[] kvalue = new double[l];
			for (i = 0; i < l; i++)
				kvalue[i] = Kernel.k_function(x, model.SV[i], model.param);

			int[] start = new int[nr_class];
			start[0] = 0;
			for (i = 1; i < nr_class; i++)
				start[i] = start[i - 1] + model.nSV[i - 1];

			int[] vote = new int[nr_class];
			for (i = 0; i < nr_class; i++)
				vote[i] = 0;

			int p = 0;
			for (i = 0; i < nr_class; i++)
				for (int j = i + 1; j < nr_class; j++) {
					double sum = 0;
					int si = start[i];
					int sj = start[j];
					int ci = model.nSV[i];
					int cj = model.nSV[j];

					int k;
					double[] coef1 = model.sv_coef[j - 1];
					double[] coef2 = model.sv_coef[i];
					for (k = 0; k < ci; k++)
						sum += coef1[si + k] * kvalue[si + k];
					for (k = 0; k < cj; k++)
						sum += coef2[sj + k] * kvalue[sj + k];
					sum -= model.rho[p];
					dec_values[p] = sum;

					if (dec_values[p] > 0)
						++vote[i];
					else
						++vote[j];
					p++;
				}

			int vote_max_idx = 0;
			for (i = 1; i < nr_class; i++)
				if (vote[i] > vote[vote_max_idx])
					vote_max_idx = i;

			return model.label[vote_max_idx];
		}
	}

	// label: label name, start: begin of each class, count: #data of classes,
	// perm: indices to the original data
	// perm, length l, must be allocated before calling this subroutine
	private static void groupClasses(SVMProblem prob, int[] nr_class_ret,
			int[][] label_ret, int[][] start_ret, int[][] count_ret, int[] perm) {
		int l = prob.l;
		int max_nr_class = 16;
		int nr_class = 0;
		int[] label = new int[max_nr_class];
		int[] count = new int[max_nr_class];
		int[] data_label = new int[l];
		int i;

		for (i = 0; i < l; i++) {
			int this_label = (int) (prob.y[i]);
			int j;
			for (j = 0; j < nr_class; j++) {
				if (this_label == label[j]) {
					++count[j];
					break;
				}
			}
			data_label[i] = j;
			if (j == nr_class) {
				if (nr_class == max_nr_class) {
					max_nr_class *= 2;
					int[] new_data = new int[max_nr_class];
					System.arraycopy(label, 0, new_data, 0, label.length);
					label = new_data;
					new_data = new int[max_nr_class];
					System.arraycopy(count, 0, new_data, 0, count.length);
					count = new_data;
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}

		//
		// Labels are ordered by their first occurrence in the training set.
		// However, for two-class sets with -1/+1 labels and -1 appears first,
		// we swap labels to ensure that internally the binary SVM has positive
		// data corresponding to the +1 instances.
		//
		if (nr_class == 2 && label[0] == -1 && label[1] == +1) {
			do {
				int _x = label[0];
				label[0] = label[1];
				label[1] = _x;
			} while (false);
			do {
				int _x = count[0];
				count[0] = count[1];
				count[1] = _x;
			} while (false);
			for (i = 0; i < l; i++) {
				if (data_label[i] == 0)
					data_label[i] = 1;
				else
					data_label[i] = 0;
			}
		}

		int[] start = new int[nr_class];
		start[0] = 0;
		for (i = 1; i < nr_class; i++)
			start[i] = start[i - 1] + count[i - 1];
		for (i = 0; i < l; i++) {
			perm[start[data_label[i]]] = i;
			++start[data_label[i]];
		}
		start[0] = 0;
		for (i = 1; i < nr_class; i++)
			start[i] = start[i - 1] + count[i - 1];

		nr_class_ret[0] = nr_class;
		label_ret[0] = label;
		start_ret[0] = start;
		count_ret[0] = count;
	}

	private String checkParameter(SVMProblem prob, SVMParameter param) {
		// svm_type

		SVMType svm_type = param.svm_type;

		// kernel_type, degree

		if (param.gamma < 0)
			return "gamma < 0";

		if (param.degree < 0)
			return "degree of polynomial kernel < 0";

		// cache_size,eps,C,nu,p,shrinking

		if (param.cache_size <= 0)
			return "cache_size <= 0";

		if (param.eps <= 0)
			return "eps <= 0";

		if (svm_type.equals(SVMType.C_SVC)
				|| svm_type.equals(SVMType.EPSILON_SVR)
				|| svm_type.equals(SVMType.NU_SVR))
			if (param.C <= 0)
				return "C <= 0";

		if (svm_type.equals(SVMType.NU_SVC)
				|| svm_type.equals(SVMType.ONE_CLASS_SVM)
				|| svm_type.equals(SVMType.NU_SVR))
			if (param.nu <= 0 || param.nu > 1)
				return "nu <= 0 or nu > 1";

		if (svm_type.equals(SVMType.EPSILON_SVR))
			if (param.p < 0)
				return "p < 0";

		if (param.probability && svm_type.equals(SVMType.ONE_CLASS_SVM))
			return "one-class SVM probability output not supported yet";

		// check whether nu-svc is feasible

		if (svm_type.equals(SVMType.NU_SVC)) {
			int l = prob.l;
			int max_nr_class = 16;
			int nr_class = 0;
			int[] label = new int[max_nr_class];
			int[] count = new int[max_nr_class];

			int i;
			for (i = 0; i < l; i++) {
				int this_label = (int) prob.y[i];
				int j;
				for (j = 0; j < nr_class; j++)
					if (this_label == label[j]) {
						++count[j];
						break;
					}

				if (j == nr_class) {
					if (nr_class == max_nr_class) {
						max_nr_class *= 2;
						int[] new_data = new int[max_nr_class];
						System.arraycopy(label, 0, new_data, 0, label.length);
						label = new_data;

						new_data = new int[max_nr_class];
						System.arraycopy(count, 0, new_data, 0, count.length);
						count = new_data;
					}
					label[nr_class] = this_label;
					count[nr_class] = 1;
					++nr_class;
				}
			}

			for (i = 0; i < nr_class; i++) {
				int n1 = count[i];
				for (int j = i + 1; j < nr_class; j++) {
					int n2 = count[j];
					if (param.nu * (n1 + n2) / 2 > Math.min(n1, n2))
						return "specified nu is infeasible";
				}
			}
		}

		return null;
	}

	/**
	 * @param data
	 * @return
	 * @throws Exception
	 */
	private Instances preProcessData(Instances data) throws Exception {
		// remove instances with missing class
		Instances insts = new Instances(data);
		insts.deleteWithMissingClass();

		if (!getDoNotReplaceMissingValues()) {
			m_ReplaceMissingValues = new ReplaceMissingValues();
			m_ReplaceMissingValues.setInputFormat(insts);
			insts = Filter.useFilter(insts, m_ReplaceMissingValues);
		}

		// can classifier handle the data?
		// we check this here so that if the user turns off
		// replace missing values filtering, it will fail
		// if the data actually does have missing values
		getCapabilities().testWithFail(insts);

		m_NormalizeFilter = null;
		if (getNormalize()) {
			m_NormalizeFilter = new Normalize();
			m_NormalizeFilter.setInputFormat(insts);
			insts = Filter.useFilter(insts, m_NormalizeFilter);
		}
		return insts;
	}

	/**
	 * @param inst
	 * @return
	 */
	private SVMNode[] instanceToArray(Instance inst) {
		int avilableAttributes = 0;
		for (int i = 0; i < inst.numAttributes(); ++i) {
			if (inst.classIndex() == i)
				continue;
			if (inst.isMissing(i))
				continue;
			++avilableAttributes;
		}

		SVMNode[] instanceData = new SVMNode[avilableAttributes];
		int counter = 0;
		for (int i = 0; i < inst.numAttributes(); ++i) {
			if (inst.classIndex() == i)
				continue;
			if (inst.isMissing(i))
				continue;
			instanceData[counter] = new SVMNode();
			instanceData[counter].index = i;
			instanceData[counter].value = inst.value(i);
			++counter;
		}
		return instanceData;
	}

	/**
	 * returns the svm_problem
	 * 
	 * @param data
	 *            the x values
	 * @param classifications
	 *            the y values
	 * @return the svm_problem object
	 */
	private SVMProblem getProblem(SVMNode[][] data, double[] classifications) {
		SVMProblem problem = new SVMProblem();
		problem.l = data.length;
		problem.x = data;
		problem.y = classifications;
		return problem;
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		if (!getDoNotReplaceMissingValues() && isMissing(instance)) {
			synchronized (m_ReplaceMissingValues) {
				m_ReplaceMissingValues.input(instance);
				m_ReplaceMissingValues.batchFinished();
				instance = m_ReplaceMissingValues.output();
			}
		}

		if (getNormalize()) {
			synchronized (m_NormalizeFilter) {
				m_NormalizeFilter.input(instance);
				m_NormalizeFilter.batchFinished();
				instance = m_NormalizeFilter.output();
			}
		}

		SVMNode[] x = instanceToArray(instance);
		double[] result = new double[instance.numClasses()];

		if (m_parameters.probability
				&& (m_parameters.svm_type.equals(SVMType.C_SVC) || m_parameters.svm_type
						.equals(SVMType.NU_SVC))) {
			int[] labels = new int[instance.numClasses()];
			if (m_Model.label != null)
				for (int i = 0; i < m_Model.nr_class; i++)
					labels[i] = m_Model.label[i];
			double[] prob_estimates = new double[instance.numClasses()];

			predictProbability(m_Model, x, prob_estimates);
			for (int k = 0; k < prob_estimates.length; k++)
				result[labels[k]] = prob_estimates[k];
		} else {
			double v = predict(m_Model, x);
			if (instance.classAttribute().isNumeric()) {
				result[0] = v;
			} else if (!m_parameters.svm_type.equals(SVMType.ONE_CLASS_SVM)) {
				result[(int) Math.round(v)] = 1;
			} else if (v > 0) {
				result[0] = 1;
			} else {
				// outlier (interface for Classifier specifies that
				// unclassified instances
				// should return a distribution of all zeros)
				result[0] = 0;
			}
		}

		return result;
	}

	private boolean isMissing(Instance instance) {
		for (int i = 0; i < instance.numAttributes(); ++i) {
			if (i == instance.classIndex())
				continue;
			if (instance.isMissing(i))
				return true;
		}
		return false;
	}

	public SVMModel getModel() {
		return m_Model;
	}

	//
	// decision_function
	//
	private static class DecisionFunction {
		double[] alpha;
		double rho;
	};

	// java: information about solution except alpha,
	// because we cannot return multiple values otherwise...
	private static class SolutionInfo {
		public double rho;
		public double r; // for Solver_NU
	}

	/**
	 * An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918 Solves:
	 * 
	 * min 0.5(\alpha^T Q \alpha) + p^T \alpha
	 * 
	 * y^T \alpha = \delta y_i = +1 or -1 0 <= alpha_i <= Cp for y_i = 1 0 <=
	 * alpha_i <= Cn for y_i = -1
	 * 
	 * Given:
	 * 
	 * Q, p, y, Cp, Cn, and an initial feasible point \alpha l is the size of
	 * vectors and matrices eps is the stopping tolerance
	 * 
	 * solution will be put in \alpha, objective value will be put in obj
	 */
	private class Solver {
		int active_size;
		byte[] y;
		double[] G; // gradient of objective function
		static final byte LOWER_BOUND = 0;
		static final byte UPPER_BOUND = 1;
		static final byte FREE = 2;
		byte[] alpha_status; // LOWER_BOUND, UPPER_BOUND, FREE
		double[] alpha;
		Kernel Q;
		double[] QD;
		double eps;
		double Cp, Cn;
		double[] p;
		int[] active_set;
		double[] G_bar; // gradient, if we treat free variables as 0
		int l;
		boolean unshrink; // XXX

		static final double INF = java.lang.Double.POSITIVE_INFINITY;

		double get_C(int i) {
			return (y[i] > 0) ? Cp : Cn;
		}

		void update_alpha_status(int i) {
			if (alpha[i] >= get_C(i))
				alpha_status[i] = UPPER_BOUND;
			else if (alpha[i] <= 0)
				alpha_status[i] = LOWER_BOUND;
			else
				alpha_status[i] = FREE;
		}

		boolean is_upper_bound(int i) {
			return alpha_status[i] == UPPER_BOUND;
		}

		boolean is_lower_bound(int i) {
			return alpha_status[i] == LOWER_BOUND;
		}

		boolean is_free(int i) {
			return alpha_status[i] == FREE;
		}

		void swap_index(int i, int j) {
			Q.swap_index(i, j);
			do {
				byte _x = y[i];
				y[i] = y[j];
				y[j] = _x;
			} while (false);
			do {
				double _x = G[i];
				G[i] = G[j];
				G[j] = _x;
			} while (false);
			do {
				byte _x = alpha_status[i];
				alpha_status[i] = alpha_status[j];
				alpha_status[j] = _x;
			} while (false);
			do {
				double _x = alpha[i];
				alpha[i] = alpha[j];
				alpha[j] = _x;
			} while (false);
			do {
				double _x = p[i];
				p[i] = p[j];
				p[j] = _x;
			} while (false);
			do {
				int _x = active_set[i];
				active_set[i] = active_set[j];
				active_set[j] = _x;
			} while (false);
			do {
				double _x = G_bar[i];
				G_bar[i] = G_bar[j];
				G_bar[j] = _x;
			} while (false);
		}

		void reconstruct_gradient() {
			// reconstruct inactive elements of G from G_bar and free variables

			if (active_size == l)
				return;

			int i, j;
			int nr_free = 0;

			for (j = active_size; j < l; j++)
				G[j] = G_bar[j] + p[j];

			for (j = 0; j < active_size; j++)
				if (is_free(j))
					nr_free++;

			if (nr_free * l > 2 * active_size * (l - active_size)) {
				for (i = active_size; i < l; i++) {
					float[] Q_i = Q.get_Q(i, active_size);
					for (j = 0; j < active_size; j++)
						if (is_free(j))
							G[i] += alpha[j] * Q_i[j];
				}
			} else {
				for (i = 0; i < active_size; i++)
					if (is_free(i)) {
						float[] Q_i = Q.get_Q(i, l);
						double alpha_i = alpha[i];
						for (j = active_size; j < l; j++)
							G[j] += alpha_i * Q_i[j];
					}
			}
		}

		void Solve(int l, Kernel Q, double[] p_, byte[] y_, double[] alpha_,
				double Cp, double Cn, double eps, SolutionInfo si,
				boolean shrinking) {
			this.l = l;
			this.Q = Q;
			QD = Q.get_QD();
			p = (double[]) p_.clone();
			y = (byte[]) y_.clone();
			alpha = (double[]) alpha_.clone();
			this.Cp = Cp;
			this.Cn = Cn;
			this.eps = eps;
			this.unshrink = false;

			// initialize alpha_status
			{
				alpha_status = new byte[l];
				for (int i = 0; i < l; i++)
					update_alpha_status(i);
			}

			// initialize active set (for shrinking)
			{
				active_set = new int[l];
				for (int i = 0; i < l; i++)
					active_set[i] = i;
				active_size = l;
			}

			// initialize gradient
			{
				G = new double[l];
				G_bar = new double[l];
				int i;
				for (i = 0; i < l; i++) {
					G[i] = p[i];
					G_bar[i] = 0;
				}
				for (i = 0; i < l; i++)
					if (!is_lower_bound(i)) {
						float[] Q_i = Q.get_Q(i, l);
						double alpha_i = alpha[i];
						int j;
						for (j = 0; j < l; j++)
							G[j] += alpha_i * Q_i[j];
						if (is_upper_bound(i))
							for (j = 0; j < l; j++)
								G_bar[j] += get_C(i) * Q_i[j];
					}
			}

			// optimization step

			int iter = 0;
			int max_iter = Math.max(10000000,
					l > Integer.MAX_VALUE / 100 ? Integer.MAX_VALUE : 100 * l);
			int counter = Math.min(l, 1000) + 1;
			int[] working_set = new int[2];

			while (iter < max_iter) {
				// show progress and do shrinking

				if (--counter == 0) {
					counter = Math.min(l, 1000);
					if (shrinking)
						do_shrinking();
				}

				if (select_working_set(working_set) != 0) {
					// reconstruct the whole gradient
					reconstruct_gradient();
					// reset active set size and check
					active_size = l;
					if (select_working_set(working_set) != 0)
						break;
					else
						counter = 1; // do shrinking next iteration
				}

				int i = working_set[0];
				int j = working_set[1];

				++iter;

				// update alpha[i] and alpha[j], handle bounds carefully

				float[] Q_i = Q.get_Q(i, active_size);
				float[] Q_j = Q.get_Q(j, active_size);

				double C_i = get_C(i);
				double C_j = get_C(j);

				double old_alpha_i = alpha[i];
				double old_alpha_j = alpha[j];

				if (y[i] != y[j]) {
					double quad_coef = QD[i] + QD[j] + 2 * Q_i[j];
					if (quad_coef <= 0)
						quad_coef = 1e-12;
					double delta = (-G[i] - G[j]) / quad_coef;
					double diff = alpha[i] - alpha[j];
					alpha[i] += delta;
					alpha[j] += delta;

					if (diff > 0) {
						if (alpha[j] < 0) {
							alpha[j] = 0;
							alpha[i] = diff;
						}
					} else {
						if (alpha[i] < 0) {
							alpha[i] = 0;
							alpha[j] = -diff;
						}
					}
					if (diff > C_i - C_j) {
						if (alpha[i] > C_i) {
							alpha[i] = C_i;
							alpha[j] = C_i - diff;
						}
					} else {
						if (alpha[j] > C_j) {
							alpha[j] = C_j;
							alpha[i] = C_j + diff;
						}
					}
				} else {
					double quad_coef = QD[i] + QD[j] - 2 * Q_i[j];
					if (quad_coef <= 0)
						quad_coef = 1e-12;
					double delta = (G[i] - G[j]) / quad_coef;
					double sum = alpha[i] + alpha[j];
					alpha[i] -= delta;
					alpha[j] += delta;

					if (sum > C_i) {
						if (alpha[i] > C_i) {
							alpha[i] = C_i;
							alpha[j] = sum - C_i;
						}
					} else {
						if (alpha[j] < 0) {
							alpha[j] = 0;
							alpha[i] = sum;
						}
					}
					if (sum > C_j) {
						if (alpha[j] > C_j) {
							alpha[j] = C_j;
							alpha[i] = sum - C_j;
						}
					} else {
						if (alpha[i] < 0) {
							alpha[i] = 0;
							alpha[j] = sum;
						}
					}
				}

				// update G

				double delta_alpha_i = alpha[i] - old_alpha_i;
				double delta_alpha_j = alpha[j] - old_alpha_j;

				for (int k = 0; k < active_size; k++) {
					G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
				}

				// update alpha_status and G_bar

				{
					boolean ui = is_upper_bound(i);
					boolean uj = is_upper_bound(j);
					update_alpha_status(i);
					update_alpha_status(j);
					int k;
					if (ui != is_upper_bound(i)) {
						Q_i = Q.get_Q(i, l);
						if (ui)
							for (k = 0; k < l; k++)
								G_bar[k] -= C_i * Q_i[k];
						else
							for (k = 0; k < l; k++)
								G_bar[k] += C_i * Q_i[k];
					}

					if (uj != is_upper_bound(j)) {
						Q_j = Q.get_Q(j, l);
						if (uj)
							for (k = 0; k < l; k++)
								G_bar[k] -= C_j * Q_j[k];
						else
							for (k = 0; k < l; k++)
								G_bar[k] += C_j * Q_j[k];
					}
				}
			}

			if (iter >= max_iter) {
				if (active_size < l) {
					// reconstruct the whole gradient to calculate objective
					// value
					reconstruct_gradient();
					active_size = l;
				}
			}

			// calculate rho

			si.rho = calculate_rho();

			// put back the solution
			{
				for (int i = 0; i < l; i++)
					alpha_[active_set[i]] = alpha[i];
			}

		}

		// return 1 if already optimal, return 0 otherwise
		int select_working_set(int[] working_set) {
			// return i,j such that
			// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
			// j: mimimizes the decrease of obj value
			// (if quadratic coefficeint <= 0, replace it with tau)
			// -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

			double Gmax = -INF;
			double Gmax2 = -INF;
			int Gmax_idx = -1;
			int Gmin_idx = -1;
			double obj_diff_min = INF;

			for (int t = 0; t < active_size; t++)
				if (y[t] == +1) {
					if (!is_upper_bound(t))
						if (-G[t] >= Gmax) {
							Gmax = -G[t];
							Gmax_idx = t;
						}
				} else {
					if (!is_lower_bound(t))
						if (G[t] >= Gmax) {
							Gmax = G[t];
							Gmax_idx = t;
						}
				}

			int i = Gmax_idx;
			float[] Q_i = null;
			if (i != -1) // null Q_i not accessed: Gmax=-INF if i=-1
				Q_i = Q.get_Q(i, active_size);

			for (int j = 0; j < active_size; j++) {
				if (y[j] == +1) {
					if (!is_lower_bound(j)) {
						double grad_diff = Gmax + G[j];
						if (G[j] >= Gmax2)
							Gmax2 = G[j];
						if (grad_diff > 0) {
							double obj_diff;
							double quad_coef = QD[i] + QD[j] - 2.0 * y[i]
									* Q_i[j];
							if (quad_coef > 0)
								obj_diff = -(grad_diff * grad_diff) / quad_coef;
							else
								obj_diff = -(grad_diff * grad_diff) / 1e-12;

							if (obj_diff <= obj_diff_min) {
								Gmin_idx = j;
								obj_diff_min = obj_diff;
							}
						}
					}
				} else {
					if (!is_upper_bound(j)) {
						double grad_diff = Gmax - G[j];
						if (-G[j] >= Gmax2)
							Gmax2 = -G[j];
						if (grad_diff > 0) {
							double obj_diff;
							double quad_coef = QD[i] + QD[j] + 2.0 * y[i]
									* Q_i[j];
							if (quad_coef > 0)
								obj_diff = -(grad_diff * grad_diff) / quad_coef;
							else
								obj_diff = -(grad_diff * grad_diff) / 1e-12;

							if (obj_diff <= obj_diff_min) {
								Gmin_idx = j;
								obj_diff_min = obj_diff;
							}
						}
					}
				}
			}

			if (Gmax + Gmax2 < eps)
				return 1;

			working_set[0] = Gmax_idx;
			working_set[1] = Gmin_idx;
			return 0;
		}

		private boolean be_shrunk(int i, double Gmax1, double Gmax2) {
			if (is_upper_bound(i)) {
				if (y[i] == +1)
					return (-G[i] > Gmax1);
				else
					return (-G[i] > Gmax2);
			} else if (is_lower_bound(i)) {
				if (y[i] == +1)
					return (G[i] > Gmax2);
				else
					return (G[i] > Gmax1);
			} else
				return (false);
		}

		void do_shrinking() {
			int i;
			double Gmax1 = -INF; // max { -y_i * grad(f)_i | i in I_up(\alpha) }
			double Gmax2 = -INF; // max { y_i * grad(f)_i | i in I_low(\alpha) }

			// find maximal violating pair first
			for (i = 0; i < active_size; i++) {
				if (y[i] == +1) {
					if (!is_upper_bound(i)) {
						if (-G[i] >= Gmax1)
							Gmax1 = -G[i];
					}
					if (!is_lower_bound(i)) {
						if (G[i] >= Gmax2)
							Gmax2 = G[i];
					}
				} else {
					if (!is_upper_bound(i)) {
						if (-G[i] >= Gmax2)
							Gmax2 = -G[i];
					}
					if (!is_lower_bound(i)) {
						if (G[i] >= Gmax1)
							Gmax1 = G[i];
					}
				}
			}

			if (unshrink == false && Gmax1 + Gmax2 <= eps * 10) {
				unshrink = true;
				reconstruct_gradient();
				active_size = l;
			}

			for (i = 0; i < active_size; i++)
				if (be_shrunk(i, Gmax1, Gmax2)) {
					active_size--;
					while (active_size > i) {
						if (!be_shrunk(active_size, Gmax1, Gmax2)) {
							swap_index(i, active_size);
							break;
						}
						active_size--;
					}
				}
		}

		double calculate_rho() {
			double r;
			int nr_free = 0;
			double ub = INF, lb = -INF, sum_free = 0;
			for (int i = 0; i < active_size; i++) {
				double yG = y[i] * G[i];

				if (is_lower_bound(i)) {
					if (y[i] > 0)
						ub = Math.min(ub, yG);
					else
						lb = Math.max(lb, yG);
				} else if (is_upper_bound(i)) {
					if (y[i] < 0)
						ub = Math.min(ub, yG);
					else
						lb = Math.max(lb, yG);
				} else {
					++nr_free;
					sum_free += yG;
				}
			}

			if (nr_free > 0)
				r = sum_free / nr_free;
			else
				r = (ub + lb) / 2;

			return r;
		}

	}

	/**
	 * Solver for nu-svm classification and regression
	 * 
	 * additional constraint: e^T \alpha = constant
	 */
	private final class Solver_NU extends Solver {
		private SolutionInfo si;

		void Solve(int l, Kernel Q, double[] p, byte[] y, double[] alpha,
				double Cp, double Cn, double eps, SolutionInfo si,
				boolean shrinking) {
			this.si = si;
			super.Solve(l, Q, p, y, alpha, Cp, Cn, eps, si, shrinking);
		}

		// return 1 if already optimal, return 0 otherwise
		int select_working_set(int[] working_set) {
			// return i,j such that y_i = y_j and
			// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
			// j: minimizes the decrease of obj value
			// (if quadratic coefficeint <= 0, replace it with tau)
			// -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

			double Gmaxp = -INF;
			double Gmaxp2 = -INF;
			int Gmaxp_idx = -1;

			double Gmaxn = -INF;
			double Gmaxn2 = -INF;
			int Gmaxn_idx = -1;

			int Gmin_idx = -1;
			double obj_diff_min = INF;

			for (int t = 0; t < active_size; t++)
				if (y[t] == +1) {
					if (!is_upper_bound(t))
						if (-G[t] >= Gmaxp) {
							Gmaxp = -G[t];
							Gmaxp_idx = t;
						}
				} else {
					if (!is_lower_bound(t))
						if (G[t] >= Gmaxn) {
							Gmaxn = G[t];
							Gmaxn_idx = t;
						}
				}

			int ip = Gmaxp_idx;
			int in = Gmaxn_idx;
			float[] Q_ip = null;
			float[] Q_in = null;
			if (ip != -1) // null Q_ip not accessed: Gmaxp=-INF if ip=-1
				Q_ip = Q.get_Q(ip, active_size);
			if (in != -1)
				Q_in = Q.get_Q(in, active_size);

			for (int j = 0; j < active_size; j++) {
				if (y[j] == +1) {
					if (!is_lower_bound(j)) {
						double grad_diff = Gmaxp + G[j];
						if (G[j] >= Gmaxp2)
							Gmaxp2 = G[j];
						if (grad_diff > 0) {
							double obj_diff;
							double quad_coef = QD[ip] + QD[j] - 2 * Q_ip[j];
							if (quad_coef > 0)
								obj_diff = -(grad_diff * grad_diff) / quad_coef;
							else
								obj_diff = -(grad_diff * grad_diff) / 1e-12;

							if (obj_diff <= obj_diff_min) {
								Gmin_idx = j;
								obj_diff_min = obj_diff;
							}
						}
					}
				} else {
					if (!is_upper_bound(j)) {
						double grad_diff = Gmaxn - G[j];
						if (-G[j] >= Gmaxn2)
							Gmaxn2 = -G[j];
						if (grad_diff > 0) {
							double obj_diff;
							double quad_coef = QD[in] + QD[j] - 2 * Q_in[j];
							if (quad_coef > 0)
								obj_diff = -(grad_diff * grad_diff) / quad_coef;
							else
								obj_diff = -(grad_diff * grad_diff) / 1e-12;

							if (obj_diff <= obj_diff_min) {
								Gmin_idx = j;
								obj_diff_min = obj_diff;
							}
						}
					}
				}
			}

			if (Math.max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < eps)
				return 1;

			if (y[Gmin_idx] == +1)
				working_set[0] = Gmaxp_idx;
			else
				working_set[0] = Gmaxn_idx;
			working_set[1] = Gmin_idx;

			return 0;
		}

		private boolean be_shrunk(int i, double Gmax1, double Gmax2,
				double Gmax3, double Gmax4) {
			if (is_upper_bound(i)) {
				if (y[i] == +1)
					return (-G[i] > Gmax1);
				else
					return (-G[i] > Gmax4);
			} else if (is_lower_bound(i)) {
				if (y[i] == +1)
					return (G[i] > Gmax2);
				else
					return (G[i] > Gmax3);
			} else
				return (false);
		}

		void do_shrinking() {
			double Gmax1 = -INF; // max { -y_i * grad(f)_i | y_i = +1, i in
									// I_up(\alpha) }
			double Gmax2 = -INF; // max { y_i * grad(f)_i | y_i = +1, i in
									// I_low(\alpha) }
			double Gmax3 = -INF; // max { -y_i * grad(f)_i | y_i = -1, i in
									// I_up(\alpha) }
			double Gmax4 = -INF; // max { y_i * grad(f)_i | y_i = -1, i in
									// I_low(\alpha) }

			// find maximal violating pair first
			int i;
			for (i = 0; i < active_size; i++) {
				if (!is_upper_bound(i)) {
					if (y[i] == +1) {
						if (-G[i] > Gmax1)
							Gmax1 = -G[i];
					} else if (-G[i] > Gmax4)
						Gmax4 = -G[i];
				}
				if (!is_lower_bound(i)) {
					if (y[i] == +1) {
						if (G[i] > Gmax2)
							Gmax2 = G[i];
					} else if (G[i] > Gmax3)
						Gmax3 = G[i];
				}
			}

			if (unshrink == false
					&& Math.max(Gmax1 + Gmax2, Gmax3 + Gmax4) <= eps * 10) {
				unshrink = true;
				reconstruct_gradient();
				active_size = l;
			}

			for (i = 0; i < active_size; i++)
				if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4)) {
					active_size--;
					while (active_size > i) {
						if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4)) {
							swap_index(i, active_size);
							break;
						}
						active_size--;
					}
				}
		}

		double calculate_rho() {
			int nr_free1 = 0, nr_free2 = 0;
			double ub1 = INF, ub2 = INF;
			double lb1 = -INF, lb2 = -INF;
			double sum_free1 = 0, sum_free2 = 0;

			for (int i = 0; i < active_size; i++) {
				if (y[i] == +1) {
					if (is_lower_bound(i))
						ub1 = Math.min(ub1, G[i]);
					else if (is_upper_bound(i))
						lb1 = Math.max(lb1, G[i]);
					else {
						++nr_free1;
						sum_free1 += G[i];
					}
				} else {
					if (is_lower_bound(i))
						ub2 = Math.min(ub2, G[i]);
					else if (is_upper_bound(i))
						lb2 = Math.max(lb2, G[i]);
					else {
						++nr_free2;
						sum_free2 += G[i];
					}
				}
			}

			double r1, r2;
			if (nr_free1 > 0)
				r1 = sum_free1 / nr_free1;
			else
				r1 = (ub1 + lb1) / 2;

			if (nr_free2 > 0)
				r2 = sum_free2 / nr_free2;
			else
				r2 = (ub2 + lb2) / 2;

			si.r = (r1 + r2) / 2;
			return (r1 - r2) / 2;
		}
	}

}
