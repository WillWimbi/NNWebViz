//-async?
//-await?
//-when run?

let data = this.data;//grab image buffer [0] and [1]

let db = mongodb.db("mnistLab");
let collection = db.collection("pretrained100");

document.getElementById("staticNetButton").addEventListener("click", async () => {
    //navigate to our second html page
    //send request to mongoDB
    //grab image indices used in training at each pointer
    //plan of images:
    //run-0:{
    //params:{the basics, yk....}
    //checkpoints:{training data at that point in time:
    //each checkpoint possesses: 
    //iter, epoch, loss, val loss, accuracy, val accuracy, {indices from raw this.trainImages 10 worst perfomers on val loss:}
    //}
    //}
    

});


