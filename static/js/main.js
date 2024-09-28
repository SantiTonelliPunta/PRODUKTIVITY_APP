$(document).ready(function() {
    // Función para mandar mensaje de bienvenida
    function sendWelcomeMessage(message) {
        $("#chat-box").append(`<div class="message assistant-message"><p>${message}</p></div>`);
        $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
    }

    // Enviar mensaje de bienvenida al cargar la pantalla
    sendWelcomeMessage("Hola, bienvenido a PRODUKTIVITY AI. ¿En qué puedo ayudarte hoy?");

    // Funcionalidad para el formulario de chat
    $("#chat-form").on("submit", function(e) {
        e.preventDefault();
        let message = $("#message").val().trim();
        if (message === "") return;

        // Mostrar mensaje del usuario
        $("#chat-box").append(`<div class="message user-message"><p>${message}</p></div>`);
        $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
        $("#message").val("");

        // Obtener tiempo inicial
        let startTime = performance.now();

        // Mostrar loader de "Dame unos segundos que estoy creando tu respuesta!"
        let loaderId = `loader-${Date.now()}`;
        $("#chat-box").append(`<div class="message assistant-message" id="${loaderId}"><p>Dame unos segundos que estoy creando tu respuesta!</p></div>`);
        $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);

        // Enviar mensaje al servidor
        $.ajax({
            url: "/chat",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({ message: message }),
            success: function(response) {
                let endTime = performance.now();
                let responseTime = ((endTime - startTime) / 1000).toFixed(1); // Calcular tiempo de respuesta

                // Reemplazar el loader con la respuesta real del asistente
                let cleanedResponse = cleanMarkdown(response.respuesta);
                let formattedResponse = addLineBreaks(cleanedResponse);

                $(`#${loaderId}`).replaceWith(`<div class="message assistant-message"><p>${formattedResponse} (Tiempo de respuesta: ${responseTime} segundos)</p></div>`);
                $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
            },
            error: function(xhr, status, error) {
                // Reemplazar el loader con un mensaje de error
                $(`#${loaderId}`).replaceWith(`<div class="message assistant-message"><p>Hubo un error al procesar tu solicitud.</p></div>`);
                $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
                console.error("Error en la solicitud AJAX:", status, error);
            }
        });
    });

    // Función para limpiar Markdown
    function cleanMarkdown(text) {
        return text.replace(/\*\*/g, '').replace(/\*/g, '');
    }

    // Función para agregar saltos de línea
    function addLineBreaks(text) {
        // Primero, reemplazamos los puntos seguidos de dos espacios o más con un salto de línea doble
        text = text.replace(/\.\s{2,}/g, '.<br><br>');
        // Luego, reemplazamos los puntos seguidos de un solo espacio con un salto de línea simple
        text = text.replace(/\.\s/g, '.<br>');
        return text;
    }

    // Funcionalidad para los botones de sugerencias
    $(".suggestion").on("click", function() {
        let suggestionText = $(this).text();
        $("#message").val(suggestionText);
        $("#message").focus();
    });
});