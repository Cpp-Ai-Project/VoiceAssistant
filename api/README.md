# API

Uses the [Google Knowledge Graph Search API](https://developers.google.com/knowledge-graph/#sample_request).

## Dependencies

[JSON in Java](https://github.com/stleary/JSON-java)

### Installation

[Include JAR in classpath or use either Maven or Gradle](https://search.maven.org/artifact/org.json/json/20180813/bundle)

## Usage

```java
public class Main {
  public static void main(String[] args) {
    API googleApi = GoogleAPI.getInstance();
    Response response = googleApi.query("Lebron James");

    System.out.println(response.getContentUrl());
    System.out.println(response.getImageUrl());
    System.out.println(response.getDescriptionUrl());
    System.out.println(response.getDetailedDescription());
    System.out.println(response.getStatus());
  }
}
```