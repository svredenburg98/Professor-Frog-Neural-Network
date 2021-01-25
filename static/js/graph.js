d3.json("/predict").then(function(data){

    var names = data.map(d => d.x);
    console.log("names", names);

    // var trace = {
    //     x: data.map(d => d.x),
    //     y: data.map(d => d.y),
    //     type: 'bar'
    // }
    
    // var data = [trace];
    
    // Plotly.newPlot("plot", data);
    
    })