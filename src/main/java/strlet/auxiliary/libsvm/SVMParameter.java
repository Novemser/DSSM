package strlet.auxiliary.libsvm;

import java.io.Serializable;

public class SVMParameter implements Cloneable, Serializable {

	private static final long serialVersionUID = -2733609912517132812L;
	
	public SVMType svm_type;
	public KernelType kernel_type;
	public int degree; // for poly
	public double gamma; // for poly/rbf/sigmoid
	public double coef0; // for poly/sigmoid

	// these are for training only
	public double cache_size; // in MB
	public double eps; // stopping criteria
	public double C; // for C_SVC, EPSILON_SVR and NU_SVR
	public int nr_weight; // for C_SVC
	public int[] weight_label; // for C_SVC
	public double[] weight; // for C_SVC
	public double nu; // for NU_SVC, ONE_CLASS, and NU_SVR
	public double p; // for EPSILON_SVR
	public boolean shrinking; // use the shrinking heuristics
	public boolean probability; // do probability estimates

	public Object clone() {
		try {
			return super.clone();
		} catch (CloneNotSupportedException e) {
			return null;
		}
	}

}
