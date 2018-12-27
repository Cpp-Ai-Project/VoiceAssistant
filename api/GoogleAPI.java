package api;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;

/**
 * Implements the API Interface. This class uses the Google Knowledge Graph
 * Search.
 */
public class GoogleAPI implements API {
    private String baseUrl = "https://kgsearch.googleapis.com/v1/entities:search?";
    private String limit = "1";
    private final String ApiKey;
    /**
     * TODO: Dyanmically change USER_AGENT to be determinted automatically
     */
    private String USER_AGENT = "Chrome/60.0";

    private GoogleAPI() {
        ApiKey = ApiKeyReader.readKey();
    }

    /**
     * Creates new instance of API and reads the necessary api key.
     * 
     * @return API the created API object.
     */
    public static API getInstance() {
        return new GoogleAPI();
    }

    /**
     * Queries the Google Knowledge Graph Search.
     * 
     * @param query the String for query (ex: Steph Curry).
     * @return the Response of the search.
     * @see Response
     */
    public Response query(String query) {
        query = createQuery(query);
        return sendGetRequest(query);
    }

    private Response sendGetRequest(String query) {
        Response response;
        try {
            URL url = new URL(query);
            HttpURLConnection con = (HttpURLConnection) url.openConnection();
            con.setRequestMethod("GET");
            con.setRequestProperty("User-Agent", USER_AGENT);

            if (con.getResponseCode() > HttpURLConnection.HTTP_BAD_REQUEST) {
                ResponseParser responseParser = new ResponseParser(con.getErrorStream());
                response = responseParser.parse();
                con.disconnect();
                System.err.println("Connection failed");

                return response;
            }

            ResponseParser responseParser = new ResponseParser(con.getInputStream());
            response = responseParser.parse();
            con.disconnect();

            return response;
        } catch (MalformedURLException mule) {
            mule.printStackTrace();
            System.err.println("Query url incorrect");
            System.exit(2);
        } catch (ProtocolException pe) {
            pe.printStackTrace();
            System.exit(2);
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }

        return new Response();
    }

    private String createQuery(String query) {
        String q = query.replaceAll(" ", "+");
        // return baseUrl +
        // String.format("limit=%1$s&query=%2$s&fields=itemListElement", limit, q);
        return baseUrl + String.format("limit=%1$s&query=%2$s&key=%3$s&fields=itemListElement", limit, q, ApiKey);
    }
}