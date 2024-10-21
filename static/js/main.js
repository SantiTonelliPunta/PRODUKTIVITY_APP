$(document).ready(function() {
    function sendWelcomeMessage(message) {
        $("#chat-box").append(`<div class="message assistant-message"><p>${message}</p></div>`);
    }

    sendWelcomeMessage("Hola Santi, bienvenido a PRODUKTIVITY AI. ¿En qué puedo ayudarte hoy?");

    $("#chat-form").on("submit", function(e) {
        e.preventDefault();
        let message = $("#message").val().trim();
        if (message === "") return;

        let chatBox = $("#chat-box");

        // Añadimos el mensaje del usuario
        chatBox.append(`<div class="message user-message"><p>${message}</p></div>`);
        $("#message").val("");

        let startTime = performance.now();

        let loaderId = `loader-${Date.now()}`;
        chatBox.append(`
            <div class="message assistant-message" id="${loaderId}" style="margin: 10px 0; padding: 10px;">
                <div class="response-content">
                    <p>Dame unos segundos que estoy creando tu respuesta...<span class="loading-dots"></span></p>
                </div>
                <div class="response-time"></div>
            </div>
        `);

        chatBox.scrollTop(chatBox[0].scrollHeight); // Desplazar al final después de añadir el loader

        // Esta es la funcion de los dots animados.
        function animateDots() {
            let dots = '';
            return setInterval(() => {
                dots = dots.length < 3 ? dots + '.' : '';
                $(`#${loaderId} .loading-dots`).text(dots);
            }, 500);
        }

        let dotsInterval = animateDots();

        function updateWaitMessage(message) {
            $(`#${loaderId} .response-content p`).html(`${message}<span class="loading-dots"></span>`);
        }

        let timers = [
            setTimeout(() => updateWaitMessage("Estoy analizando varias opciones."), 4000),
            setTimeout(() => updateWaitMessage("Vamos a darle otro enfoque."), 8000),
            setTimeout(() => updateWaitMessage("Esta es una pregunta compleja."), 11000),
            setTimeout(() => updateWaitMessage("Lo siento pero se me está haciendo muy largo, esto no es normal"), 18000)
        ];

        // Solicitud AJAX
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

                // Añadimos la respuesta
                $(`#${loaderId}`).css({'margin-top': '10px', 'margin-bottom': '10px', 'padding': '15px'}).find('.response-content').html(formattedResponse);

                // Añadimos el tiempo de respuesta
                $(`#${loaderId}`).find('.response-time').text(`Tiempo de respuesta: ${responseTime} segundos`);

                // Desplazar al inicio del nuevo mensaje para que se vea desde el principio
                let newMessage = document.getElementById(loaderId);
                newMessage.scrollIntoView({ behavior: 'smooth', block: 'start' });
            },
            error: function(xhr, status, error) {
                timers.forEach(clearTimeout);
                clearInterval(dotsInterval);

                $(`#${loaderId} .response-content`).html(`<p>Hubo un error al procesar tu solicitud.</p>`);
                $(`#${loaderId} .response-time`).text('');

                // Desplazar al inicio del nuevo mensaje para que se vea desde el principio
                let newMessage = document.getElementById(loaderId);
                newMessage.scrollIntoView({ behavior: 'smooth', block: 'start' });

                console.error("Error en la solicitud AJAX:", status, error);
            }
        });
    });

    function cleanMarkdown(text) {
        return text.replace(/\*\*/g, '').replace(/\*/g, '');
    }

    function addLineBreaks(text) {
        text = text.replace(/\.\s{2,}/g, '.<br><br>');
        text = text.replace(/\.\s/g, '.<br>');
        text = text.replace(/(<br>|^)([-\d]+\.\s)/gm, '$1&nbsp;&nbsp;$2');
        return text;
    }

    $(".suggestion").on("click", function() {
        let suggestionText = $(this).text();
        $("#message").val(suggestionText);
        $("#message").focus();
    });
});