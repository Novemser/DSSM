package strlet.transferLearning.inductive.taskLearning.svm;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.Set;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import strlet.auxiliary.libsvm.KernelType;
import strlet.auxiliary.libsvm.LibSVM;
import strlet.auxiliary.libsvm.SVMModel;
import strlet.auxiliary.libsvm.SVMNode;
import strlet.auxiliary.libsvm.SVMParameter;
import strlet.auxiliary.libsvm.SVMProblem;
import strlet.auxiliary.libsvm.SVMType;
import strlet.auxiliary.libsvm.kernel.Kernel;
import strlet.auxiliary.libsvm.kernel.SVC_Q;
import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.taskLearning.SingleSourceModelTransfer;

public class ASVM extends SingleSourceModelTransfer {

	/** the different kernel types */
	public static final Tag[] TAGS_KERNELTYPE = {
			new Tag(KernelType.LINEAR.ordinal(), "linear: u'*v"),
			new Tag(KernelType.POLYNOMIAL.ordinal(),
					"polynomial: (gamma*u'*v + coef0)^degree"),
			new Tag(KernelType.RBF.ordinal(),
					"radial basis function: exp(-gamma*|u-v|^2)"),
			new Tag(KernelType.SIGMOID.ordinal(),
					"sigmoid: tanh(gamma*u'*v + coef0)") };

	private KernelType m_kernel_type = KernelType.LINEAR;
	private int m_degree = 3; // for poly
	private double m_Gamma = 0; // for poly/rbf/sigmoid
	private double m_coef0 = 0; // for poly/sigmoid
	private double m_Cost = 1;

	/** The filter used to get rid of missing values. */
	private Filter m_ReplaceMissingValues;

	private SVMModel m_Model;

	/**
	 * Sets type of kernel function (default KERNELTYPE_RBF)
	 * 
	 * @param value
	 *            the kernel type
	 */
	public void setKernelType(SelectedTag value) {
		if (value.getTags() == TAGS_KERNELTYPE) {
			m_kernel_type = KernelType.values()[value.getSelectedTag().getID()];
		}
	}

	/**
	 * Gets type of kernel function
	 * 
	 * @return the kernel type
	 */
	public SelectedTag getKernelType() {
		return new SelectedTag(m_kernel_type.ordinal(), TAGS_KERNELTYPE);
	}

	/**
	 * Sets the degree of the kernel
	 * 
	 * @param value
	 *            the degree of the kernel
	 */
	public void setDegree(int value) {
		m_degree = value;
	}

	/**
	 * Gets the degree of the kernel
	 * 
	 * @return the degree of the kernel
	 */
	public int getDegree() {
		return m_degree;
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
		m_coef0 = value;
	}

	/**
	 * Gets coef
	 * 
	 * @return the coef
	 */
	public double getCoef0() {
		return m_coef0;
	}

	public void setCost(double cost) {
		m_Cost = cost;
	}

	public double getCost() {
		return m_Cost;
	}

	@Override
	protected void buildModel(Instances source) throws Exception {

		source = new Instances(source);
		source.deleteWithMissingClass();
		LibSVM svm = new LibSVM();
		svm.setKernelType(new SelectedTag(m_kernel_type.ordinal(),
				LibSVM.TAGS_KERNELTYPE));
		svm.setDegree(m_degree);
		svm.setGamma(m_Gamma);
		svm.setCoef0(m_coef0);
		svm.setCost(m_Cost);
		svm.setSVMType(new SelectedTag(SVMType.C_SVC.ordinal(),
				LibSVM.TAGS_SVMTYPE));
		svm.buildClassifier(source);
		m_Model = svm.getModel();

	}

	@Override
	protected void transferModel(Instances target) throws Exception {
		m_Model = transferModel(m_Model, target);
	}

	private SVMModel transferModel(SVMModel orig, Instances insts)
			throws Exception {

		m_ReplaceMissingValues = new ReplaceMissingValues();
		m_ReplaceMissingValues.setInputFormat(insts);
		insts = Filter.useFilter(insts, m_ReplaceMissingValues);

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

		SVMParameter parameters = initializeParameters();
		if (getGamma() == 0) {
			int counter = 0;
			for (boolean b : used) {
				if (b) {
					++counter;
				}
			}
			parameters.gamma = 1.0 / counter;
		} else {
			parameters.gamma = m_Gamma;
		}

		SVMProblem prob = getProblem(instancesData, classifications);
		String error_msg = checkParameter(prob, parameters);
		if (error_msg != null)
			throw new Exception("Error: " + error_msg);

		// group training data of the same class
		int[] labels = getLables(prob.y);
		int nr_class = labels.length;
		SVMNode[][][] x = groupClasses(prob);

		// calculate weighted C
		double[] weighted_C = new double[nr_class];
		for (int i = 0; i < nr_class; ++i) {
			weighted_C[i] = parameters.C;
		}
		for (int i = 0; i < parameters.nr_weight; ++i) {
			int j;
			for (j = 0; j < nr_class; ++j) {
				if (parameters.weight_label[i] == labels[j]) {
					break;
				}
			}
			if (j == nr_class) {
				System.err.println("warning: class label "
						+ parameters.weight_label[i]
						+ "specified in weight is not found");
			} else {
				weighted_C[j] *= parameters.weight[i];
			}
		}

		// train k*(k-1)/2 models
		boolean[][] nonZero = new boolean[x.length][];
		for (int i = 0; i < nonZero.length; ++i) {
			nonZero[i] = new boolean[x[i].length];
			for (int j = 0; j < nonZero[i].length; ++j) {
				nonZero[i][j] = false;
			}
		}
		DecisionFunction[] f = new DecisionFunction[nr_class * (nr_class - 1)
				/ 2];

		int p = 0;
		for (int i = 0; i < nr_class; ++i) {
			for (int j = i + 1; j < nr_class; ++j) {
				SVMProblem sub_prob = new SVMProblem();
				int labelI = labels[i];
				int labelJ = labels[j];
				int ci = x[i].length;
				int cj = x[j].length;

				sub_prob.l = ci + cj;
				sub_prob.x = new SVMNode[sub_prob.l][];
				sub_prob.y = new double[sub_prob.l];
				for (int k = 0; k < ci; ++k) {
					sub_prob.x[k] = x[i][k];
					sub_prob.y[k] = +1;
				}
				for (int k = 0; k < cj; ++k) {
					sub_prob.x[ci + k] = x[j][k];
					sub_prob.y[ci + k] = -1;
				}

				// get auxiliary values
				double[] aux_fx = calcAux(orig, sub_prob, labelI, labelJ);

				f[p] = adapt_svm_train_one(sub_prob, parameters, aux_fx,
						weighted_C[i], weighted_C[j]);
				for (int k = 0; k < ci; ++k) {
					if (!nonZero[i][k] && (Math.abs(f[p].alpha[k]) > 0)) {
						nonZero[i][k] = true;
					}
				}
				for (int k = 0; k < cj; ++k) {
					if (!nonZero[j][k] && (Math.abs(f[p].alpha[ci + k]) > 0)) {
						nonZero[j][k] = true;
					}
				}
				++p;
			}
		}

		// build output
		SVMModel model = new SVMModel();
		model.param = parameters;
		model.nr_class = nr_class;
		model.label = labels;
		model.rho = new double[nr_class * (nr_class - 1) / 2];
		p = 0;
		for (int i = 0; i < nr_class; ++i) {
			for (int j = i + 1; j < nr_class; ++j) {
				int labelI = labels[i];
				int labelJ = labels[j];

				model.rho[p] = f[p].rho;
				model.rho[p] += origRho(orig, labelI, labelJ);
				++p;
			}
		}

		// probabilistic output not supported yet
		model.probA = null;
		model.probB = null;

		int total_sv = 0;
		int[] nz_count = new int[nr_class];
		model.nSV = new int[nr_class];
		for (int i = 0; i < nr_class; ++i) {
			int nSV = 0;
			for (boolean b : nonZero[i]) {
				if (b) {
					++nSV;
				}
			}
			model.nSV[i] = nSV;
			nz_count[i] = nSV;
			total_sv += nSV;
		}

		model.l = total_sv;
		model.SV = new SVMNode[total_sv][];
		p = 0;
		for (int i = 0; i < nonZero.length; ++i) {
			for (int j = 0; j < nonZero[i].length; ++j) {
				if (nonZero[i][j]) {
					model.SV[p++] = x[i][j];
				}
			}
		}

		int[] nz_start = new int[nr_class];
		nz_start[0] = 0;
		for (int i = 1; i < nr_class; ++i) {
			nz_start[i] = nz_start[i - 1] + nz_count[i - 1];
		}

		model.sv_coef = new double[nr_class - 1][];
		for (int i = 0; i < nr_class - 1; ++i) {
			model.sv_coef[i] = new double[total_sv];
		}

		p = 0;
		for (int i = 0; i < nr_class; ++i) {
			for (int j = i + 1; j < nr_class; ++j) {
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int ci = x[i].length;
				int cj = x[j].length;

				int q = nz_start[i];
				int k;
				for (k = 0; k < ci; k++) {
					if (nonZero[i][k]) {
						model.sv_coef[j - 1][q++] = f[p].alpha[k];
					}
				}
				q = nz_start[j];
				for (k = 0; k < cj; k++) {
					if (nonZero[j][k]) {
						model.sv_coef[i][q++] = f[p].alpha[ci + k];
					}
				}
				++p;
			}
		}

		return model;
	}

	private double[] calcAux(SVMModel orig, SVMProblem sub_prob, int labelI,
			int labelJ) {

		double[] aux_fx = new double[sub_prob.l];
		for (int i = 0; i < aux_fx.length; ++i) {
			aux_fx[i] = 1;
		}

		// Find the labels in the original problem
		int a = -1;
		int b = -1;
		for (int i = 0; i < orig.nr_class; ++i) {
			if (orig.label[i] == labelI) {
				a = i;
			} else if (orig.label[i] == labelJ) {
				b = i;
			}
		}

		// Do the labels exist in original problem
		if ((a == -1) || (b == -1))
			return aux_fx;

		// add or subtract
		int scorePred = (a < b) ? -1 : 1;
		// which of the k*(k-1) values is the right one?
		int min = Math.min(a, b);
		int max = Math.max(a, b);
		int location = min * orig.nr_class;
		location -= min * (1 + min) / 2;
		location += max - min - 1;

		for (int k = 0; k < aux_fx.length; ++k) {
			double[] dec_values = predictValues(orig, sub_prob.x[k]);
			// The binary class assumption
			double score = dec_values[location];
			aux_fx[k] += (scorePred * score * sub_prob.y[k]);
		}
		return aux_fx;

	}

	private double origRho(SVMModel orig, int labelI, int labelJ) {

		// Find the labels in the original problem
		int a = -1;
		int b = -1;
		for (int i = 0; i < orig.nr_class; ++i) {
			if (orig.label[i] == labelI) {
				a = i;
			} else if (orig.label[i] == labelJ) {
				b = i;
			}
		}

		// Do the labels exist in original problem
		if ((a == -1) || (b == -1))
			return 0;

		// which of the k*(k-1) values is the right one?
		int min = Math.min(a, b);
		int max = Math.max(a, b);
		int location = min * orig.nr_class;
		location -= min * (1 + min) / 2;
		location += max - min - 1;

		double rho = orig.rho[location];
		if (a < b) {
			return rho;
		} else {
			return 0 - rho;
		}

	}

	private SVMNode[][][] groupClasses(SVMProblem prob) {

		int[] label = getLables(prob.y);
		int[] mapping = new int[1 + label[Utils.maxIndex(label)]];
		for (int i = 0; i < label.length; ++i) {
			mapping[label[i]] = i;
		}
		int[] count = new int[label.length];
		for (double d : prob.y) {
			int tmp = (int) Math.round(d);
			tmp = mapping[tmp];
			++count[tmp];
		}

		SVMNode[][][] retVal = new SVMNode[count.length][][];
		for (int i = 0; i < count.length; ++i) {
			retVal[i] = new SVMNode[count[i]][];
			count[i] = 0;
		}

		for (int i = 0; i < prob.l; ++i) {
			double d = prob.y[i];
			int tmp = (int) Math.round(d);
			tmp = mapping[tmp];
			retVal[tmp][count[tmp]] = prob.x[i];
			++count[tmp];
		}
		return retVal;

	}

	private int[] getLables(double[] labels) {

		LinkedList<Integer> list = new LinkedList<Integer>();
		Set<Integer> set = new HashSet<Integer>();
		for (double d : labels) {
			Integer tmp = (int) Math.round(d);
			if (set.add(tmp)) {
				list.addLast(tmp);
			}
		}

		int[] arr = new int[list.size()];
		for (int i = 0; i < arr.length; ++i) {
			arr[i] = list.get(i);
		}
		return arr;

	}

	private DecisionFunction adapt_svm_train_one(SVMProblem prob,
			SVMParameter param, double[] aux_fx, double Cpos, double Cneg) {

		int l = prob.l;
		double[] alpha = new double[l];

		// initialize the model parameters
		byte[] y = new byte[l];
		for (int i = 0; i < l; ++i) {
			alpha[i] = 0;
			if (prob.y[i] > 0) {
				y[i] = +1;
			} else {
				y[i] = -1;
			}
		}

		// Solve optimization problem
		AdaptSolver s = new AdaptSolver(l);
		SVC_Q svc = new SVC_Q(prob, param, y);
		double solution = s.Solve(svc, aux_fx, y, alpha, Cpos, Cneg, param.eps,
				param.shrinking);

		// after the model is trained, we let alpha to absorb y,
		// so that we don't need to save both alpha and y in the model file
		for (int i = 0; i < l; ++i) {
			alpha[i] *= y[i];
		}

		DecisionFunction f = new DecisionFunction();
		f.alpha = alpha;
		f.rho = solution;
		return f;

	}

	private double[] predictValues(SVMModel model, SVMNode[] x) {

		int nr_class = model.nr_class;
		double[] dec_values = new double[nr_class * (nr_class - 1) / 2];
		int l = model.l;

		double[] kvalue = new double[l];
		for (int i = 0; i < l; ++i) {
			kvalue[i] = Kernel.k_function(x, model.SV[i], model.param);
		}

		int[] start = new int[nr_class];
		start[0] = 0;
		for (int i = 1; i < nr_class; ++i) {
			start[i] = start[i - 1] + model.nSV[i - 1];
		}

		int p = 0;
		int pos = 0;
		for (int i = 0; i < nr_class; ++i) {
			for (int j = i + 1; j < nr_class; ++j) {
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model.nSV[i];
				int cj = model.nSV[j];

				double[] coef1 = model.sv_coef[j - 1];
				double[] coef2 = model.sv_coef[i];
				for (int k = 0; k < ci; ++k) {
					sum += coef1[si + k] * kvalue[si + k];
				}
				for (int k = 0; k < cj; ++k) {
					sum += coef2[sj + k] * kvalue[sj + k];
				}
				sum -= model.rho[p++];
				dec_values[pos++] = sum;
			}
		}
		return dec_values;

	}

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

	private SVMProblem getProblem(SVMNode[][] data, double[] classifications) {

		SVMProblem problem = new SVMProblem();
		problem.l = data.length;
		problem.x = data;
		problem.y = classifications;
		return problem;

	}

	private SVMParameter initializeParameters() {

		SVMParameter parameters = new SVMParameter();
		parameters.svm_type = SVMType.C_SVC;
		parameters.kernel_type = KernelType.values()[new SelectedTag(
				m_kernel_type.ordinal(), LibSVM.TAGS_KERNELTYPE)
				.getSelectedTag().getID()];
		parameters.degree = m_degree;
		parameters.gamma = m_Gamma;
		parameters.coef0 = m_coef0;
		parameters.C = m_Cost;
		parameters.nu = 0.5;
		parameters.cache_size = 40;
		parameters.eps = 1e-3;
		parameters.p = 0.1;
		parameters.shrinking = true;
		parameters.nr_weight = 0;
		parameters.probability = false;
		parameters.weight = new double[0];
		parameters.weight_label = new int[0];
		return parameters;

	}

	private String checkParameter(SVMProblem prob, SVMParameter param) {

		// svm_type
		SVMType svm_type = param.svm_type;

		// kernel_type, degree
		KernelType kernel_type = param.kernel_type;
		if (!kernel_type.equals(KernelType.LINEAR) && param.gamma < 0)
			return "gamma < 0";
		if (kernel_type.equals(KernelType.POLYNOMIAL) && param.degree < 0)
			return "degree of polynomial kernel < 0";

		// cache_size,eps,C,nu,p,shrinking
		if (param.cache_size <= 0)
			return "cache_size <= 0";
		if (param.eps <= 0)
			return "eps <= 0";

		if (svm_type.equals(SVMType.C_SVC)) {
			if (param.C <= 0)
				return "C <= 0";
		} else {
			return "Unsupported SVM type";
		}

		if (param.probability && svm_type.equals(SVMType.ONE_CLASS_SVM))
			return "one-class SVM probability output not supported yet";

		return null;
	}

	@Override
	public SingleSourceTransfer makeDuplicate() throws Exception {

		ASVM dup = new ASVM();
		dup.setCoef0(m_coef0);
		dup.setCost(m_Cost);
		dup.setDegree(m_degree);
		dup.setGamma(m_Gamma);
		dup.m_kernel_type = m_kernel_type;
		if (m_Model != null) {
			throw new Exception("Cannot duplicate base model");
		}
		return dup;
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		if (isMissing(instance)) {
			synchronized (m_ReplaceMissingValues) {
				m_ReplaceMissingValues.input(instance);
				m_ReplaceMissingValues.batchFinished();
				instance = m_ReplaceMissingValues.output();
			}
		}

		SVMNode[] x = instanceToArray(instance);
		double v = predict(m_Model, x);
		double[] result = new double[instance.numClasses()];
		result[(int) Math.round(v)] = 1.0;
		return result;
	}

	private double predict(SVMModel model, SVMNode[] x) {

		int nr_class = model.nr_class;
		double[] dec_values = predictValues(model, x);

		int[] vote = new int[nr_class];
		for (int i = 0; i < nr_class; ++i) {
			vote[i] = 0;
		}
		int pos = 0;
		for (int i = 0; i < nr_class; ++i) {
			for (int j = i + 1; j < nr_class; ++j) {
				if (dec_values[pos++] > 0) {
					++vote[i];
				} else {
					++vote[j];
				}
			}
		}

		int vote_max_idx = 0;
		for (int i = 1; i < nr_class; ++i) {
			if (vote[i] > vote[vote_max_idx]) {
				vote_max_idx = i;
			}
		}
		return model.label[vote_max_idx];

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

	//
	// decision_function
	//
	private static class DecisionFunction {
		double[] alpha;
		double rho;
	};

	private static class AdaptSolver {

		private static final char LOWER_BOUND = 0;
		private static final char UPPER_BOUND = 1;
		private static final char FREE = 2;

		private final int l;

		private int active_size;
		private byte[] y;
		private double[] G; // gradient of objective function
		private char[] alpha_status; // LOWER_BOUND, UPPER_BOUND, FREE
		private double[] alpha;
		private Kernel Q;
		private double eps;
		private double Cp, Cn;
		private double[] aux_fx;

		private int[] active_set;
		private double[] G_bar; // gradient, if we treat free variables as 0
		private boolean unshrinked; // XXX

		public AdaptSolver(int l_) {
			l = l_;
		}

		private double get_C(int i) {
			return (y[i] > 0) ? Cp : Cn;
		}

		private void update_alpha_status(int i) {
			if (alpha[i] >= get_C(i)) {
				alpha_status[i] = UPPER_BOUND;
			} else if (alpha[i] <= 0) {
				alpha_status[i] = LOWER_BOUND;
			} else {
				alpha_status[i] = FREE;
			}
		}

		private boolean is_upper_bound(int i) {
			return alpha_status[i] == UPPER_BOUND;
		}

		private boolean is_lower_bound(int i) {
			return alpha_status[i] == LOWER_BOUND;
		}

		private boolean is_free(int i) {
			return alpha_status[i] == FREE;
		}

		// Jun Yang: we change the original arugment "b" to "lambda" for
		// adaptation
		public double Solve(SVC_Q svc, double[] aux_fx_, byte[] labels,
				double[] alpha_, double cPos, double cNeg, double eps,
				boolean shrinking) {

			this.Q = svc;
			y = (byte[]) labels.clone();
			aux_fx = (double[]) aux_fx_.clone();
			alpha = (double[]) alpha_.clone();
			this.Cp = cPos;
			this.Cn = cNeg;
			this.eps = eps;
			unshrinked = false;

			// initialize alpha_status whether each alpha is upperbound or
			// lowerbound
			alpha_status = new char[l];
			for (int i = 0; i < l; ++i) {
				update_alpha_status(i);
			}

			// initialize active set (for shrinking)
			// at beginning, every instance is in the active set
			active_set = new int[l];
			for (int i = 0; i < l; ++i) {
				active_set[i] = i;
			}
			active_size = l;

			// initialize gradient
			// G_i = L_D '(alpha_i) = yi*f(xi) - 1 = - aux_fx_i + y_i *
			// \sum_j(alpha_j*y_j) + y_i \sum_j (alpha_j * y_j * K_ij)
			// aux_fx = 1 - y_i * f^a(x_i)
			// note this L_D is the negative of the L_D appearing in typical SVM
			// derivation, and this L_D is to be minimized
			G = new double[l];
			G_bar = new double[l];
			for (int i = 0; i < l; ++i) {
				G[i] = -aux_fx[i];
				G_bar[i] = 0;
			}
			for (int i = 0; i < l; ++i) {
				// if alpha_i = 0, i has no contribution to G
				if (!is_lower_bound(i)) {
					// a row of the Q matrix (Q_ij = y_i * y_j * K_ij)
					float[] Q_i = Q.get_Q(i, l);
					double alpha_i = alpha[i];
					for (int j = 0; j < l; ++j) {
						// alpha_i*Q_i[j];
						G[j] += alpha_i * (Q_i[j] + y[j] * y[i]);
					}
					if (is_upper_bound(i)) {
						for (int j = 0; j < l; ++j) {
							// get_C(i) * Q_i[j];
							G_bar[j] += get_C(i) * (Q_i[j] + y[j] * y[i]);
						}
					}
				}
			}

			// optimization step
			int counter = Math.min(5 * l, 1000) + 1;
			while (true) {
				// show progress and do shrinking when the current counter
				// expires
				// it doesn't stop the iteration since a new counter is set
				if (--counter == 0) {
					counter = Math.min(5 * l, 1000);
					if (shrinking) {
						do_shrinking();
					}
					// info("."); info_flush();
				}

				// if cannot find a working variable to improve the obj function
				// (when return = 1),
				// we reconstruct the gradient and try again; if fails agin,
				// quick, otherwise,
				// continue and do shrinkage in next iteration
				int[] i_ = { 0 };
				if (!select_working_set(i_)) {
					// reconstruct the whole gradient
					reconstruct_gradient();
					// reset active set size and check
					active_size = l;
					// info("*"); info_flush();
					if (!select_working_set(i_)) {
						break;
					} else {
						// do shrinking next iteration
						counter = 1;
					}
				}

				int i = i_[0];

				// update alpha[i] and handle bounds carefully
				float[] Q_i = Q.get_Q(i, active_size);
				double C_i = get_C(i);
				double old_alpha_i = alpha[i];
				alpha[i] = old_alpha_i - (G[i] / (Q_i[i] + 1));

				if (alpha[i] < 0) {
					alpha[i] = 0;
				}
				if (alpha[i] > C_i) {
					alpha[i] = C_i;
				}

				double delta_alpha_i = alpha[i] - old_alpha_i;

				// update G
				for (int k = 0; k < active_size; ++k) {
					G[k] += delta_alpha_i * (Q_i[k] + y[k] * y[i]);
				}

				// update alpha_status and G_bar
				// read the old status
				boolean ui = is_upper_bound(i);
				update_alpha_status(i);

				int k;
				// upper bound status changes for i
				if (ui != is_upper_bound(i)) {
					Q_i = Q.get_Q(i, l);
					if (ui) {
						// upper bound -> no upper bound
						for (k = 0; k < l; k++) {
							G_bar[k] -= C_i * (Q_i[k] + y[k] * y[i]);
						}
					} else {
						// no upper bound -> upper bound
						for (k = 0; k < l; k++) {
							G_bar[k] += C_i * (Q_i[k] + y[k] * y[i]);
						}
					}
				}
			}

			// calculate threshold (only happens after all SMO iterations)
			// = calculate_rho();
			double sol = 0;
			for (int k = 0; k < l; ++k) {
				sol -= alpha[k] * y[k];
			}

			// put back the solution
			for (int i = 0; i < l; ++i) {
				alpha_[active_set[i]] = alpha[i];
			}

			return sol;

		}

		private void reconstruct_gradient() {

			// quit if there is no inactive elements
			if (active_size == l) {
				return;
			}

			// only the Gi for inactive elements i has been modified,
			// the Gi for active elements remain the same
			// traverse inactive elements
			for (int i = active_size; i < l; ++i) {
				G[i] = G_bar[i] - aux_fx[i];
			}

			for (int i = 0; i < active_size; ++i) {
				if (is_free(i)) {
					float[] Q_i = Q.get_Q(i, l);
					double alpha_i = alpha[i];
					for (int j = active_size; j < l; j++)
						// traverse inactive elements
						G[j] += alpha_i * (Q_i[j] + y[i] * y[j]);
				}
			}

		}

		private boolean select_working_set(int[] out) {

			double Gmin = 0;
			double Gmax = 0;
			int Gmin_idx = -1;
			int Gmax_idx = -1;

			for (int t = 0; t < active_size; t++) {
				if (!is_upper_bound(t)) {
					if (G[t] <= Gmin) {
						Gmin = G[t];
						Gmin_idx = t;
					}
				}

				if (!is_lower_bound(t)) {
					if (G[t] >= Gmax) {
						Gmax = G[t];
						Gmax_idx = t;
					}
				}
			}

			if (Gmax - Gmin < eps)
				return false;

			if (Math.abs(Gmax) >= Math.abs(Gmin))
				out[0] = Gmax_idx;
			else
				out[0] = Gmin_idx;
			return true;

		}

		private void do_shrinking() {

			int[] arr = { 0, 0 };
			if (max_violating_pair(arr) != 0) {
				return;
			}
			int i = arr[0];
			int j = arr[1];

			double Gmin = 0;
			if (i != -1) {
				Gmin = G[i];
			}
			double Gmax = 0;
			if (j != -1) {
				Gmax = G[j];
			}

			for (int k = 0; k < active_size; ++k) {
				if (is_lower_bound(k)) {
					// if(G[k] <= Gmin)
					if (G[k] <= Gmax)
						continue; // continue if active
				} else if (is_upper_bound(k)) {
					// if(G[k] >= Gmax)
					if (G[k] >= Gmin)
						continue;
				} else
					continue;

				// if runs to this point, k must be inactive, so we swap it
				// outside active set
				--active_size;
				swap_index(k, active_size);
				--k; // look at the newcomer
			}

			// if -(Gm1 + Gm2) = m - M >= 10*eps ==> m <= M + 10*eps
			// ==> reconstrunct the gradient to increase accuracy
			if (unshrinked || Gmax - Gmin > eps * 10)
				return;

			unshrinked = true;
			reconstruct_gradient();

			for (int k = l - 1; k >= active_size; --k) {
				if (is_lower_bound(k)) {
					// if(-G[k] > Gmin) continue; //continue if inactive
					if (G[k] > Gmax) {
						continue; // continue if inactive
					}
				} else if (is_upper_bound(k)) {
					// if(G[k] < Gmax) continue; //continue if inactive
					if (G[k] < Gmin) {
						continue; // continue if inactive
					}
				} else {
					continue;
				}

				swap_index(k, active_size);
				active_size++;
				++k; // look at the newcomer
			}

		}

		private int max_violating_pair(int[] arr) {

			double Gmin = 0;
			double Gmax = 0;
			int out_i = -1;
			int out_j = -1;

			for (int t = 0; t < active_size; t++) {
				if (!is_upper_bound(t)) {
					if (G[t] <= Gmin) {
						Gmin = G[t];
						out_i = t;
					}
				}

				if (!is_lower_bound(t)) {
					if (G[t] >= Gmax) {
						Gmax = G[t];
						out_j = t;
					}
				}
			}
			arr[0] = out_i;
			arr[1] = out_j;

			if (Gmax - Gmin < eps)
				return 1;

			return 0;

		}

		private void swap_index(int i, int j) {
			Q.swap_index(i, j);
			swap(aux_fx, i, j);
			swap(y, i, j);
			swap(G, i, j);
			swap(alpha_status, i, j);
			swap(alpha, i, j);
			swap(active_set, i, j);
			swap(G_bar, i, j);
		}

		private void swap(int[] arr, int i, int j) {
			int tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
		}

		private void swap(char[] arr, int i, int j) {
			char tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
		}

		private void swap(byte[] arr, int i, int j) {
			byte tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
		}

		private void swap(double[] arr, int i, int j) {
			double tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
		}
	}

}
