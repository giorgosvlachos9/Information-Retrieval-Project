import java.io.*;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Scanner;
import java.util.function.UnaryOperator;

// tested for lucene 7.7.3 and jdk13
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.Similarity;

public class Phase1 {

    private final static String READ_DIR = "IR2024\\documents.txt";
    private final static String QUERIES_DIR = "IR2024\\queries.txt";
    private final static String RESULTS_DIR = "IR2024\\trec_eval\\myResults.txt";

    public static void main(String[] args) throws IOException, ParseException {

        //  Specify the analyzer and the similarity
        Analyzer analyzer = new EnglishAnalyzer();
        Similarity similarity = new ClassicSimilarity();
        // Create the index
        String indexLocation = ("index");
        Directory index = FSDirectory.open(Paths.get(indexLocation));

        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setSimilarity(similarity);
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);

        IndexWriter writer = new IndexWriter(index, config);

        // Read the txt file
        String docs_file = ReadEntireFileIntoAString(READ_DIR);
        // Split the documents of the txt file          ---- DOCUMENTS ----
        String[] docs = txtSplitter(docs_file, "///");

        for (int i=0; i < docs.length; i++) {
            // Parse each document and return it as a DocTuple object
            DocTuple document = parseDocument(docs[i]);
            addDoc(writer, document);
        }
        writer.close();

        // Search for queries
        String queries_file = ReadEntireFileIntoAString(QUERIES_DIR);               // ---- QUERIES ----
        String[] queries = editQueries(txtSplitter(queries_file, "///"), "Q\\d+", "");

        IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexLocation)));
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new ClassicSimilarity());

        String qCode = "";
        // Create buffered writer for the myResults file
        BufferedWriter fileWriter = new BufferedWriter(new FileWriter(RESULTS_DIR));
        for (int i=0; i < queries.length; i++) {
            int temp = i + 1;
            qCode = (i < 9) ? "Q0" + temp : "Q" + temp;

            search(analyzer, searcher, "content", queries[i], qCode,50, fileWriter);
        }
        fileWriter.close();
        reader.close();


    }

    /**          -------------------- parseDocument function --------------------
     *
     * @param doc : The document we want to parse in String form
     * @return : A DocTuple object after successfully parsing the document
     */
    private static DocTuple parseDocument(String doc) {

        String code = doc.substring(0, 8);

        if (code.matches(".*[a-zA-Z0-9\\s].*")) {
            code = code.replaceAll("[a-zA-Z\\s]", " ");
            code = code.trim();
        }
        DocTuple dc = new DocTuple(code, doc);

        return dc;

    }


    /**         -------------------- addDoc function --------------------
     *
     * @param writer : IndexWriter for out index that adds the documents to it
     * @param dc : A DocTuple object that helps us get the document code and its content to define the fields
     *          of the document and add it to the index
     * @throws IOException
     */
    private static void addDoc(IndexWriter writer, DocTuple dc) throws IOException {
        Document doc = new Document();
        // Create fields
        TextField code = new TextField("code", dc.getCode(), Field.Store.YES);

        TextField content = new TextField("content", dc.getText(), Field.Store.NO);

        // Add fields to document
        doc.add(code);
        doc.add(content);
        // Add document to the index
        writer.addDocument(doc);
    }

    /**          -------------------- search function --------------------
     *
     * @param analyzer : The analyzer for our index
     * @param indexSearcher : Our index searcher
     * @param field : The field of our index to run the query on
     * @param qCode : The code of our query
     * @param searchQuery : The query we want to search for results
     * @param filewriter : The writer for our myResults.txt file
     */
    private static void search(Analyzer analyzer, IndexSearcher indexSearcher, String field, String searchQuery, String qCode,
                                int noDocs, BufferedWriter filewriter) {

        try{
            // create a query parser on the field "contents"
            QueryParser parser = new QueryParser(field, analyzer);

            // parse the query according to QueryParser
            Query query = parser.parse(searchQuery);
            System.out.println("Searching for: " + query.toString(field));

            // search the index using the indexSearcher
            TopDocs results = indexSearcher.search(query, noDocs);
            ScoreDoc[] hits = results.scoreDocs;

            // Write each element of the data array to the file
            for (int i=0; i<hits.length; i++) {
                Document hitDoc = indexSearcher.doc(hits[i].doc);
                String line = qCode + " 0 " + hitDoc.get("code") + " 0 " + hits[i].score + " myIRMethod";
                filewriter.write(line);
                filewriter.newLine(); // Add a newline after each line if needed
            }
            System.out.println("Successfull search and write!");


        } catch(Exception e){
            e.printStackTrace();
        }

    }

    /**          -------------------- ReadEntireFileIntoAString function --------------------
     * Gets a file path and retuns the same file as a String object
     */
    private static String ReadEntireFileIntoAString(String file) throws FileNotFoundException {

        Scanner scanner = new Scanner(new File(file));
        scanner.useDelimiter("\\A"); //\\A stands for :start of a string
        String entireFileText = scanner.next();
        return entireFileText;
    }

    /**          -------------------- txtSplitter function --------------------
     *
     * @param txtfile : Stringified txt file
     * @param splitregex : Regular expression to perform the split on the file
     * @return : The string array after splitting the txt file
     */
    private static String[] txtSplitter(String txtfile, String splitregex) {
        return txtfile.split(splitregex);
    }

    /**          -------------------- editQueries function --------------------
     *
     * @param array : Array of queries to edit
     * @param regex : Regular expression to replace
     * @param replacement : Replacement String for the expression
     * @return : returns the array after successfully replacing the regex with the wanted String
     */
    private static String[] editQueries(String[] array, String regex, String replacement) {
        UnaryOperator<String> replaceRegex = s -> s.replaceAll(regex, replacement);
        Arrays.setAll(array, i -> replaceRegex.apply(array[i]));
        return array;
    }

}