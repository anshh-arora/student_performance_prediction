document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('prediction-result');

    form.addEventListener('submit', function (event) {
        event.preventDefault();
        resultText.textContent = "Processing...";

        const formData = new FormData(form);
        const jsonData = {};
        formData.forEach((value, key) => (jsonData[key] = value));

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(jsonData),
        })
            .then(response => response.json())
            .then(data => {
                if (data.prediction) {
                    resultText.textContent = `Predicted Final Marks: ${data.prediction}`;
                } else {
                    resultText.textContent = `Error: ${data.error}`;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                resultText.textContent = 'An error occurred while processing your request.';
            });
    });
});
