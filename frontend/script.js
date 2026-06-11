const imageInput = document.getElementById('imageInput');
const submitBtn = document.getElementById('submitBtn');
const previewImg = document.getElementById('previewImg');
const loading = document.getElementById('loading');
const vocabOutput = document.getElementById('vocabOutput');

// Hiển thị ảnh ngay khi chọn file
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        previewImg.src = URL.createObjectURL(file);
        previewImg.classList.remove('hidden');
    }
});

// Khi bấm nút Phân tích
submitBtn.addEventListener('click', async () => {
    const file = imageInput.files[0];
    if (!file) {
        alert("Vui lòng chọn một bức ảnh trước!");
        return;
    }

    // Gói file thành form data để gửi đi
    const formData = new FormData();
    formData.append("file", file);

    // Hiển thị loading, xóa kết quả cũ
    loading.classList.remove('hidden');
    submitBtn.disabled = true;
    vocabOutput.innerHTML = '';

    try {
        // GỌI API FASTAPI CỦA BẠN
        const response = await fetch("/api/v1/predict", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        
        // Kiểm tra và hiển thị kết quả
        if (result.status === "success" && result.data.length > 0) {
            result.data.forEach(item => {
                const li = document.createElement('li');
                li.style.listStyle = "none";
                
                // Thêm thẻ img và class bọc ngoài để căn chỉnh
                li.innerHTML = `
                    <div class="vocab-card flex-card">
                        <img src="data:image/jpeg;base64,${item.cropped_image}" alt="${item.word}" class="cropped-img">
                        <div class="vocab-info">
                            <h4 class="vocab-word">🎯 ${item.word.toUpperCase()}</h4>
                            <div class="vocab-related">🔗 Từ liên quan: ${item.related.join(', ')}</div>
                            <div class="vocab-example">📝 Ví dụ: "${item.example}"</div>
                        </div>
                    </div>
                `;
                vocabOutput.appendChild(li);
            });
        } else {
            vocabOutput.innerHTML = '<li class="empty-msg">😢 AI không tìm thấy vật thể nào rõ ràng trong ảnh này.</li>';
        }
    // MỚI - hiện lỗi chi tiết hơn
    } catch (error) {
        console.error("Lỗi gọi API:", error);
        alert(`Lỗi: ${error.name} - ${error.message}`);
    } finally {
        loading.classList.add('hidden');
        submitBtn.disabled = false;
    }
});