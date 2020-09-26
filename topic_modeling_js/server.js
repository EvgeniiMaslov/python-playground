
var app = require("express")(); 
var bodyParser = require("body-parser"); 
var express = require("express"); 
//Set view engine to ejs
app.set("view engine", "ejs"); 

//Tell Express where we keep our index.ejs
app.set("views", __dirname + "/views"); 

//Use body-parser
app.use(bodyParser.urlencoded({ extended: false })); 

var default_list = [[{"term":NaN, "probability":1}]]
app.get("/", (req, res) => { res.render("index", {newListItems:default_list}); });
app.use(express.static(__dirname + '/public'));
app.post('/', function(req, res) {
    var text = req.body.input;
    var lda = require('lda');
    console.log(text);
    var documents = text.match( /[^\.!\?]+[\.!\?]+/g );
        
    var result = lda(documents, 1, 5);
    console.log(result);
    var name = result;
    res.render("index", {newListItems:name});
});


app.listen(8080, () => { console.log("Server online on http://localhost:8080"); });