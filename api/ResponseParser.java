package api;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.json.JSONTokener;

/**
 * Class to parse result returned from Google Knowledge Graph Search API. Note:
 * This class may be changed in the future if parsing does not yield good
 * results for general results or other API providers are implemented.
 * Dependencies: org.json (include with either Gradle, Maven or JAR)
 */
public class ResponseParser {
    private Response response;
    private InputStream inputStream;

    /**
     * Create parser for parsing the JSON result from Google Knowledge Graph Search
     * API.
     * 
     * @param inputStream an InputStream containing the JSON object.
     */
    public ResponseParser(InputStream inputStream) {
        this.response = new Response();
        this.inputStream = inputStream;
    }

    private void detailedDescription(BufferedReader reader) {
        try {
            String line = reader.readLine();
            while (line != null) {
                if (line.contains("detailedDescription")) {
                    JSONTokener tk = new JSONTokener(reader.readLine());
                    tk.nextTo(":");
                    response.setDetailedDescription(tk.nextTo("\0").replaceAll("^: +\"", "").replaceAll(" +\",$", ""));
                    tk = new JSONTokener(reader.readLine());
                    tk.nextTo(":");
                    response.setDescriptionUrl(tk.nextTo("\0").replaceAll("^: +\"", "").replaceAll("\",$", ""));
                    break;
                }
                line = reader.readLine();
            }
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.err.println("error during parsing for detailedDescription");
        }
    }

    private void imageUrl(BufferedReader reader) {
        try {
            String line = reader.readLine();
            while (line != null) {
                if (line.contains("image")) {
                    JSONTokener tk = new JSONTokener(reader.readLine());
                    tk.nextTo(":");
                    response.setImageUrl(tk.nextTo("\0").replaceAll("^: +\"", "").replaceAll("\",$", ""));
                    break;
                }
                line = reader.readLine();
            }
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.err.println("error during parsing for detailedDescription");
        }
    }

    private void contentUrl(BufferedReader reader) {
        try {
            String line = reader.readLine();
            while (line != null) {
                if (line.contains("url")) {
                    JSONTokener tk = new JSONTokener(line);
                    tk.nextTo(":");
                    response.setContentUrl(tk.nextTo("\0").replaceAll("^: +\"", "").replaceAll("\"$", ""));
                    break;
                }
                line = reader.readLine();
            }
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.err.println("error during parsing for detailedDescription");
        }
    }

    /**
     * 
     * @return the Reposnse of search API result
     * @see Response
     */
    public Response parse() {
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(this.inputStream));

        try {
            bufferedReader.readLine();
            imageUrl(bufferedReader);
            detailedDescription(bufferedReader);
            contentUrl(bufferedReader);
            bufferedReader.close();
        } catch (IOException ioe) {
            System.err.println("Could not read response");
        }
        response.setStatus("Success");

        return this.response;
    }
}
