package strlet.experiments;

import java.io.File;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class NewsgroupsExperiment extends TextExperiment {

	private static final String[] top = { "rec", "talk", "sci" };

	public NewsgroupsExperiment(String a, String b) {
		super(a, b);
	}

	@Override
	protected boolean isClass(String name, String classA) {
		return name.startsWith(classA);
	}

	@Override
	protected LinkedList<String> getFileNames(int fold) throws Exception {
		if ((fold < 0) || (fold >= DOMAINS)) {
			throw new Exception(
					"More domains attempted than currently supported");
		}

		ZipFile zip = new ZipFile(ZIP_FILE);
		Enumeration<? extends ZipEntry> entries = zip.entries();
		HashMap<String, Integer> subjects1 = new HashMap<String, Integer>();
		HashMap<String, Integer> subjects2 = new HashMap<String, Integer>();
		while (entries.hasMoreElements()) {
			ZipEntry entry = entries.nextElement();
			if (entry.isDirectory()) {
				String name = entry.getName().substring(0,
						entry.getName().length() - 1);
				if (name.startsWith(a)) {
					subjects1.put(name, subjects1.size());
				}
				if (name.startsWith(b)) {
					subjects2.put(name, subjects2.size());
				}
			}
		}

		entries = zip.entries();
		LinkedList<String> fileNames = new LinkedList<String>();
		while (entries.hasMoreElements()) {
			ZipEntry entry = entries.nextElement();
			if (entry.isDirectory()) {
				continue;
			}
			String name = entry.getName().substring(0,
					entry.getName().indexOf("/"));
			if (name.startsWith(a)) {
				if ((subjects1.get(name) + fold) % DOMAINS == 0) {
					fileNames.add(entry.getName());
				}
			}
			if (name.startsWith(b)) {
				if ((subjects2.get(name) + fold) % DOMAINS == 0) {
					fileNames.add(entry.getName());
				}
			}
		}
		zip.close();
		return fileNames;
	}

	public static void main(String[] args) throws Exception {
		TextExperiment.ZIP_FILE = new File("testData" + File.separator
				+ "20_newsgroups_processed.zip");
		for (String a : top) {
			for (String b : top) {
				if (a.equals(b))
					continue;
				NewsgroupsExperiment experiment = new NewsgroupsExperiment(a, b);
				experiment.runExperiment();
			}
		}
	}

}
