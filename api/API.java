package api;

/**
 * Interface for Search APIs. All implementations should have a static
 * getInstance() to create the API object and setup any API keys needed.
 */
public interface API {
    static API getInstance() {
        return null;
    }

    Response query(String query);
}
