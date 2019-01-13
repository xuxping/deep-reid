 // 从行人监测结果中选择要查询的图片
 function changeSearchImg(obj) {
    // 将图片转成流
    var path = obj.src;
    var filename = path.substring(path.lastIndexOf("/")+1, path.length);
    var data = {
        "filename": filename,
    };
    $.ajax({
        url: "/changeImg",
        type: "POST",
        data: data,
        success: function (result) {
//            eval()
            console.log(result)
            var data = JSON.parse(result);
            $("#result").html("");
            var len = data.result_urls_p.length;
            var html = "";
            for(var i = 0; i < len; i++) {
                html += "<img src='" + data.result_urls_p[i] + "'></img>";
            }
            $("#result").html(html);
        }
    });
}