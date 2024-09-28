$(document).ready(function() {
    function cleanMarkdown(text) {
        return text.replace(/\*\*/g, '').replace(/\*/g, '');
    }

    function addLineBreaks(text) {
        // Primero, reemplazamos los puntos seguidos de dos espacios o más con un salto de línea doble
        text = text.replace(/\.\s{2,}/g, '.<br><br>');
        // Luego, reemplazamos los puntos seguidos de un solo espacio con un salto de línea simple
        text = text.replace(/\.\s/g, '.<br>');
        return text;
    }

    function sendWelcomeMessage(message) {
        $("#chat-box").append(`<div class="message assistant-message"><p>${message}</p></div>`);
        $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
    }

    sendWelcomeMessage("Hola, bienvenido a PRODUKTIVITY AI. ¿En qué puedo ayudarte hoy?");

    $("#chat-form").on("submit", function(e) {
        e.preventDefault();
        let message = $("#message").val().trim();
        if (message === "") return;

        $("#chat-box").append(`<div class="message user-message"><p>${message}</p></div>`);
        $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
        $("#message").val("");

        let startTime = performance.now();

        let loaderId = `loader-${Date.now()}`;
        $("#chat-box").append(`<div class="message assistant-message" id="${loaderId}"><p>Dame unos segundos que estoy creando tu respuesta!</p></div>`);
        $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);

        $.ajax({
            url: "/chat",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({ message: message }),
            success: function(response) {
                let endTime = performance.now();
                let responseTime = ((endTime - startTime) / 1000).toFixed(1);

                let cleanedResponse = cleanMarkdown(response.respuesta);
                let formattedResponse = addLineBreaks(cleanedResponse);

                console.log("Respuesta original:", response.respuesta);
                console.log("Respuesta formateada:", formattedResponse);

                $(`#${loaderId}`).replaceWith(`<div class="message assistant-message"><p>${formattedResponse} (Tiempo de respuesta: ${responseTime} segundos)</p></div>`);
                $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
            },
            error: function(xhr, status, error) {
                $(`#${loaderId}`).replaceWith(`<div class="message assistant-message"><p>Hubo un error al procesar tu solicitud.</p></div>`);
                $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
                console.error("Error en la solicitud AJAX:", status, error);
            }
        });
    });
});