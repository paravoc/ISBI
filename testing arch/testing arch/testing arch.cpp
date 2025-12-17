#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <direct.h>
#include <algorithm>
#include <cmath>
#include <filesystem>

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace chrono;
namespace fs = filesystem;

// Структура для хранения результатов теста
struct TestResult {
    string test_name;
    int faces_found;
    long detect_time_ms;
    long embedding_time_ms;
    long total_time_ms;
    double max_similarity;
    string verdict;
};

// Структура для эталонного лица
struct ReferenceFace {
    string filename;
    vector<float> embedding;
    long embedding_time_ms;
};

// Получение эмбеддинга лица
vector<float> getFaceEmbedding(Net& arcface, const Mat& faceImage, long& elapsed_ms) {
    auto start = high_resolution_clock::now();

    // Конвертируем в RGB и ресайзим
    Mat rgb;
    cvtColor(faceImage, rgb, COLOR_BGR2RGB);
    resize(rgb, rgb, Size(112, 112), 0, 0, INTER_CUBIC);

    // Нормализация для ArcFace
    Mat blob = blobFromImage(rgb, 1.0 / 128.0, Size(112, 112),
        Scalar(127.5, 127.5, 127.5), true, false);

    arcface.setInput(blob);
    Mat embeddingMat = arcface.forward();

    auto end = high_resolution_clock::now();
    elapsed_ms = duration_cast<milliseconds>(end - start).count();

    // Конвертируем в вектор float
    vector<float> embedding(
        embeddingMat.ptr<float>(),
        embeddingMat.ptr<float>() + embeddingMat.total()
    );

    return embedding;
}

// Косинусное сходство
double cosineSimilarity(const vector<float>& a, const vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0;
    }

    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    normA = sqrt(normA);
    normB = sqrt(normB);

    if (normA > 1e-10 && normB > 1e-10) {
        return dot / (normA * normB);
    }

    return 0.0;
}

// Загрузка всех эталонных лиц из папки
vector<ReferenceFace> loadReferenceFaces(Net& detector, Net& arcface, const string& folderPath) {
    vector<ReferenceFace> references;

    cout << "Загрузка эталонных лиц из папки: " << folderPath << endl;

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            string filename = entry.path().string();
            string ext = entry.path().extension().string();

            // Поддерживаемые форматы
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                cout << "  Обработка: " << entry.path().filename().string() << "... ";

                Mat img = imread(filename);
                if (img.empty()) {
                    cout << "ОШИБКА загрузки" << endl;
                    continue;
                }

                // Детекция лица
                Mat blob = blobFromImage(img, 1.0, Size(300, 300),
                    Scalar(104, 177, 123), false, false);
                detector.setInput(blob);
                Mat detection = detector.forward();

                const float* data = detection.ptr<float>();
                bool faceFound = false;
                Rect faceRect;

                for (int i = 0; i < detection.size[2]; i++) {
                    float confidence = data[i * 7 + 2];
                    if (confidence > 0.5) {
                        int x1 = static_cast<int>(data[i * 7 + 3] * img.cols);
                        int y1 = static_cast<int>(data[i * 7 + 4] * img.rows);
                        int x2 = static_cast<int>(data[i * 7 + 5] * img.cols);
                        int y2 = static_cast<int>(data[i * 7 + 6] * img.rows);

                        // Корректировка границ
                        x1 = max(0, x1); y1 = max(0, y1);
                        x2 = min(img.cols - 1, x2); y2 = min(img.rows - 1, y2);

                        if (x2 > x1 && y2 > y1) {
                            faceRect = Rect(x1, y1, x2 - x1, y2 - y1);
                            faceFound = true;
                            break;
                        }
                    }
                }

                if (!faceFound) {
                    cout << "ОШИБКА: лицо не найдено" << endl;
                    continue;
                }

                // Извлечение эмбеддинга
                long embeddingTime;
                Mat faceImg = img(faceRect).clone();
                vector<float> embedding = getFaceEmbedding(arcface, faceImg, embeddingTime);

                ReferenceFace ref;
                ref.filename = entry.path().filename().string();
                ref.embedding = embedding;
                ref.embedding_time_ms = embeddingTime;

                references.push_back(ref);
                cout << "УСПЕХ (" << embeddingTime << " мс)" << endl;
            }
        }
    }

    cout << "Загружено эталонных лиц: " << references.size() << endl;
    return references;
}

// Сравнение с множеством эталонов
pair<double, string> compareWithReferences(const vector<float>& testEmbedding,
    const vector<ReferenceFace>& references) {
    if (references.empty() || testEmbedding.empty()) {
        return { 0.0, "Нет эталонов" };
    }

    double maxSimilarity = 0.0;
    string bestMatch = "";

    for (const auto& ref : references) {
        double similarity = cosineSimilarity(testEmbedding, ref.embedding);

        if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            bestMatch = ref.filename;
        }
    }

    return { maxSimilarity * 100.0, bestMatch }; // В процентах
}

// Получение вердикта на основе сходства
string getVerdict(double similarity) {
    if (similarity > 60.0) {
        return "Даня";
    }
    else if (similarity > 40.0) {
        return "Возможно";
    }
    else {
        return "Другой";
    }
}

int main() {
    setlocale(LC_ALL, "ru");
    cout << "=== СИСТЕМА РАСПОЗНАВАНИЯ ЛИЦ С МНОГИМИ ЭТАЛОНАМИ ===" << endl << endl;

    // 1. ЗАГРУЗКА МОДЕЛЕЙ
    cout << "1. Загрузка моделей..." << endl;
    Net detector = readNetFromCaffe("res/deploy.prototxt",
        "res/res10_300x300_ssd_iter_140000.caffemodel");
    Net arcface = readNetFromONNX("res/arcface.onnx");

    if (detector.empty() || arcface.empty()) {
        cout << "ОШИБКА: Не удалось загрузить модели!" << endl;
        return -1;
    }
    cout << "   ✓ Модели загружены" << endl;

    // 2. ЗАГРУЗКА МНОГИХ ЭТАЛОННЫХ ЛИЦ
    cout << "\n2. Загрузка эталонных лиц..." << endl;
    vector<ReferenceFace> references = loadReferenceFaces(detector, arcface, "source_photos");

    if (references.empty()) {
        cout << "ОШИБКА: Не загружено ни одного эталонного лица!" << endl;
        cout << "Проверьте папку 'source_photos/' с файлами .jpg/.png" << endl;
        return -1;
    }

    // 3. СОЗДАНИЕ ПАПКИ ДЛЯ РЕЗУЛЬТАТОВ
    string output_folder = "multi_reference_results";
    _mkdir(output_folder.c_str());

    // 4. ТЕСТИРОВАНИЕ НА 9 ФОТОГРАФИЯХ
    cout << "\n3. Тестирование на 9 фотографиях..." << endl;
    cout << string(100, '=') << endl;
    cout << left << setw(15) << "Тест"
        << setw(8) << "Лиц"
        << setw(12) << "Детекция"
        << setw(15) << "Эмбеддинг"
        << setw(12) << "Всего"
        << setw(12) << "Сходство"
        << setw(25) << "Вердикт/Эталон" << endl;
    cout << string(100, '=') << endl;

    vector<TestResult> results;

    for (int testNum = 1; testNum <= 9; testNum++) {
        string filename = "photos/test " + to_string(testNum) + ".png";
        TestResult result;
        result.test_name = "Тест " + to_string(testNum);

        auto totalStart = high_resolution_clock::now();

        // Загрузка тестового изображения
        Mat testImg = imread(filename);
        if (testImg.empty()) {
            cout << left << setw(15) << result.test_name
                << "ОШИБКА: Файл не найден!" << endl;
            continue;
        }

        // Детекция лиц
        auto detectStart = high_resolution_clock::now();
        Mat blob = blobFromImage(testImg, 1.0, Size(300, 300),
            Scalar(104, 177, 123), false, false);
        detector.setInput(blob);
        Mat detection = detector.forward();
        auto detectEnd = high_resolution_clock::now();
        result.detect_time_ms = duration_cast<milliseconds>(detectEnd - detectStart).count();

        const float* data = detection.ptr<float>();
        result.faces_found = 0;
        result.max_similarity = 0.0;
        long totalEmbeddingTime = 0;

        // Обработка каждого лица
        for (int i = 0; i < detection.size[2]; i++) {
            float confidence = data[i * 7 + 2];

            if (confidence > 0.5) {
                result.faces_found++;

                int x1 = static_cast<int>(data[i * 7 + 3] * testImg.cols);
                int y1 = static_cast<int>(data[i * 7 + 4] * testImg.rows);
                int x2 = static_cast<int>(data[i * 7 + 5] * testImg.cols);
                int y2 = static_cast<int>(data[i * 7 + 6] * testImg.rows);

                // Корректировка границ
                x1 = max(0, x1); y1 = max(0, y1);
                x2 = min(testImg.cols - 1, x2); y2 = min(testImg.rows - 1, y2);

                if (x2 > x1 && y2 > y1) {
                    Rect faceRect(x1, y1, x2 - x1, y2 - y1);

                    // Проверяем минимальный размер
                    if (faceRect.width < 20 || faceRect.height < 20) {
                        result.faces_found--; // Не считаем слишком маленькие
                        continue;
                    }

                    // Извлечение эмбеддинга
                    long embeddingTime;
                    Mat faceImg = testImg(faceRect).clone();
                    vector<float> embedding = getFaceEmbedding(arcface, faceImg, embeddingTime);
                    totalEmbeddingTime += embeddingTime;

                    // Сравнение со всеми эталонами
                    auto [similarity, bestMatch] = compareWithReferences(embedding, references);

                    if (similarity > result.max_similarity) {
                        result.max_similarity = similarity;
                    }

                    // Отрисовка результата
                    string verdict = getVerdict(similarity);
                    Scalar color;

                    if (verdict == "Даня") {
                        color = Scalar(0, 255, 0);
                    }
                    else if (verdict == "Возможно") {
                        color = Scalar(0, 200, 255);
                    }
                    else {
                        color = Scalar(0, 0, 255);
                    }

                    string label = format("%s %.1f%%", verdict.c_str(), similarity);
                    rectangle(testImg, faceRect, color, 2);
                    putText(testImg, label, Point(faceRect.x, faceRect.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                }
            }
        }

        result.embedding_time_ms = totalEmbeddingTime;

        auto totalEnd = high_resolution_clock::now();
        result.total_time_ms = duration_cast<milliseconds>(totalEnd - totalStart).count();

        // Определение общего вердикта
        if (result.faces_found == 0) {
            result.verdict = "Нет лиц";
        }
        else {
            result.verdict = getVerdict(result.max_similarity);
            result.verdict += format(" (%.0f%%)", result.max_similarity);
        }

        // Вывод в таблицу
        cout << left << setw(15) << result.test_name
            << setw(8) << result.faces_found
            << setw(12) << result.detect_time_ms << "мс"
            << setw(15) << result.embedding_time_ms << "мс"
            << setw(12) << result.total_time_ms << "мс"
            << setw(12) << fixed << setprecision(1) << result.max_similarity << "%"
            << setw(25) << result.verdict << endl;

        // Сохранение результата
        string result_file = output_folder + "/test_" + to_string(testNum) + ".jpg";
        imwrite(result_file, testImg);

        results.push_back(result);
    }

    // 5. СТАТИСТИКА И АНАЛИЗ
    cout << "\n" << string(100, '=') << endl;
    cout << "СТАТИСТИКА ТЕСТИРОВАНИЯ:" << endl;
    cout << string(100, '-') << endl;

    int total_tests = results.size();
    int total_faces = 0;
    long total_detect_time = 0;
    long total_embed_time = 0;
    long total_time = 0;
    int correct_matches = 0;

    // Ожидаемые результаты (по вашему описанию)
    vector<bool> should_be_me = { true, true, false, true, false, false, false, true, false };

    for (size_t i = 0; i < results.size(); i++) {
        if (i < should_be_me.size()) {
            bool expected_me = should_be_me[i];
            bool detected_me = (results[i].max_similarity > 60.0);

            if (expected_me == detected_me) {
                correct_matches++;
            }
        }

        total_faces += results[i].faces_found;
        total_detect_time += results[i].detect_time_ms;
        total_embed_time += results[i].embedding_time_ms;
        total_time += results[i].total_time_ms;
    }

    cout << "• Всего тестов: " << total_tests << endl;
    cout << "• Всего лиц найдено: " << total_faces << endl;
    cout << "• Корректных распознаваний: " << correct_matches << "/" << total_tests
        << " (" << fixed << setprecision(1) << (correct_matches * 100.0 / total_tests) << "%)" << endl;
    cout << "• Эталонных лиц в базе: " << references.size() << endl;
    cout << "• Среднее время детекции: " << total_detect_time / total_tests << " мс" << endl;
    cout << "• Среднее время эмбеддинга на лицо: "
        << (total_faces > 0 ? total_embed_time / total_faces : 0) << " мс" << endl;

    // 6. ДИАГНОСТИКА ЭТАЛОННЫХ ВЕКТОРОВ
    cout << "\nДИАГНОСТИКА ЭТАЛОННЫХ ВЕКТОРОВ:" << endl;
    cout << string(100, '-') << endl;

    if (references.size() >= 2) {
        // Проверяем сходство между эталонами
        for (size_t i = 0; i < references.size(); i++) {
            for (size_t j = i + 1; j < references.size(); j++) {
                double similarity = cosineSimilarity(references[i].embedding,
                    references[j].embedding) * 100.0;
                cout << "• " << references[i].filename << " ↔ " << references[j].filename
                    << ": " << fixed << setprecision(1) << similarity << "%" << endl;
            }
        }

        // Проверяем сходство эталонов самих с собой
        cout << "\nСАМОСХОДСТВО ЭТАЛОНОВ:" << endl;
        for (const auto& ref : references) {
            double self_similarity = cosineSimilarity(ref.embedding, ref.embedding) * 100.0;
            cout << "• " << ref.filename << ": " << fixed << setprecision(1)
                << self_similarity << "% (должно быть ~100%)" << endl;
        }
    }

    cout << "\n" << string(100, '=') << endl;
    cout << "РЕЗУЛЬТАТЫ СОХРАНЕНЫ В ПАПКЕ: " << output_folder << "/" << endl;

    // 7. РЕКОМЕНДАЦИИ
    cout << "\nРЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:" << endl;
    cout << string(100, '-') << endl;

    if (references.size() < 3) {
        cout << "⚠️  Добавьте больше эталонных фото (3-5 разных)" << endl;
        cout << "   • Разное освещение" << endl;
        cout << "   • Разные эмоции" << endl;
        cout << "   • Разный ракурс (но фронтально)" << endl;
    }

    // Проверяем качество эталонов
    bool good_quality = true;
    for (const auto& ref : references) {
        if (ref.embedding_time_ms > 500) {
            cout << "⚠️  Эталон " << ref.filename << " обрабатывается долго ("
                << ref.embedding_time_ms << " мс)" << endl;
            good_quality = false;
        }
    }

    if (good_quality) {
        cout << "✅ Качество эталонных фото хорошее" << endl;
    }

    cout << "\nДля выхода нажмите любую клавишу..." << endl;
    waitKey(0);

    return 0;
}