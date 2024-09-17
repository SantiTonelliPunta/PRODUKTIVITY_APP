// static/js/scripts.js

$(document).ready(function() {
    $("#chat-form").on("submit", function(e) {
        e.preventDefault();
        let message = $("#message").val().trim();
        if (message === "") return;

        // Mostrar mensaje del usuario
        $("#chat-box").append(`<div class="message user-message"><p>${message}</p></div>`);
        $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
        $("#message").val("");

        // Enviar mensaje al servidor
        $.ajax({
            url: "/chat",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({ message: message }),
            success: function(response) {
                $("#chat-box").append(`<div class="message assistant-message"><p>${response.respuesta}</p></div>`);
                $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
            },
            error: function(xhr, status, error) {
                $("#chat-box").append(`<div class="message assistant-message"><p>Hubo un error al procesar tu solicitud.</p></div>`);
                $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
            }
        });
    });
});