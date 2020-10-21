let validateInput = (event) => {
    console.log(event);
};

$('select.dropdown').dropdown();

$('#submit-form').form({
    fields: {
        sequences: 'empty',
    },
    onSuccess(event, fields){
        event.preventDefault();
        let errorMessage = $('.job.submission.messages');
        errorMessage.text("");

        fetch("/api/embeddings", {
            method: 'post',
            body: new FormData(event.target),
        }).then(response => {
            return response.json();
        }).then(response => {
            if(response.message){
                $('#submit-form').form('add prompt', 'sequences', null);
                errorMessage.text(response.message);
            } else {
                $('#id').val(response.job_id)
            }
        }).catch(e => {
            errorMessage.text("There was an error processing you request: " + e);
            console.error(e)
        })
    }
});

$('#check-form').form({
    fields: {
        id: 'empty',
    },
    onSuccess(event, fields) {
        event.preventDefault();

        let jobIdCard = $('#job-id-status-card');
        let jobStatus = $('#job-id-status-message');
        let jobDownloadOptions = $('#job-id-download-options');

        jobIdCard.removeClass('show');
        jobDownloadOptions.removeClass('show');

        fetch("/api/embeddings/status?id=" + fields.id, {
            method: 'get'
        }).then(response => {
            return response.json();
        }).then(response => {
            if(response.message){
            } else {
                jobIdCard.addClass('show');
                jobStatus.text(response.status);
                if(response.status === "SUCCESS"){
                    $("#download-embeddings").attr("href", "/api/embeddings?id=" + fields.id);
                    jobDownloadOptions.addClass('show');
                }
            }
        }).catch(e => {
            console.error(e)
        })
    }
});