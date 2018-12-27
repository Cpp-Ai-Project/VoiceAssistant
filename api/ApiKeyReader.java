package api;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Utility class to read API key from file
 */
public class ApiKeyReader {
    /**
     * Returns a String of from the API key. API key file should be named
     * "apikey.txt" and located in source root directory or directory of Java
     * entrypoint.
     *
     * @return the string in apikey.txt
     */
    public static String readKey() {
        String key = "";
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader("apikey.txt"));
            key += bufferedReader.readLine();
            bufferedReader.close();
            return key;
        } catch (FileNotFoundException fnf) {
            fnf.printStackTrace();
            System.err.println("Missing API key file. File should be placed in root and called apikey.txt");
            System.exit(1);
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.err.println("Could not read from API key file.");
            System.exit(1);
        }
        return key;
    }
}
