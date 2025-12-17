#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <chrono>
#include <direct.h>   

using namespace cv;
using namespace std;
using namespace chrono;

int main() {
    setlocale(LC_ALL, "ru");

     
    string output_folder = "output_test_caffemodel";
    auto check = _mkdir(output_folder.c_str());

    dnn::Net detector = dnn::readNetFromCaffe("res/deploy.prototxt",
        "res/res10_300x300_ssd_iter_140000.caffemodel");

    string images[] = { "photos/test 1.png", "photos/test 2.png",
                        "photos/test 3.png", "photos/test 4.png" };

    for (int i = 0; i < 4; i++) {
        cout << "\n=== Обработка: " << images[i] << " ===" << endl;

        Mat img = imread(images[i]);
        if (img.empty()) {
            cout << "ОШИБКА: Файл не найден!" << endl;
            continue;
        }

        cout << "Размер: " << img.cols << "x" << img.rows << " пикселей" << endl;

        string original_name = output_folder + "/original_test_" + to_string(i + 1) + ".jpg";
        imwrite(original_name, img);
        cout << "Оригинал: " << original_name << endl;

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        string gray_name = output_folder + "/gray_test_" + to_string(i + 1) + ".jpg";
        imwrite(gray_name, gray);
        cout << "Серое: " << gray_name << endl;

        Mat equalized;
        equalizeHist(gray, equalized);
        string eq_name = output_folder + "/equalized_test_" + to_string(i + 1) + ".jpg";
        imwrite(eq_name, equalized);
        cout << "Выравненное: " << eq_name << endl;

        Mat edges;
        Canny(gray, edges, 50, 150);
        string edges_name = output_folder + "/edges_test_" + to_string(i + 1) + ".jpg";
        imwrite(edges_name, edges);
        cout << "Контуры: " << edges_name << endl;

        Mat blur_img;
        GaussianBlur(img, blur_img, Size(5, 5), 0);
        string blur_name = output_folder + "/blur_test_" + to_string(i + 1) + ".jpg";
        imwrite(blur_name, blur_img);
        cout << "Размытие: " << blur_name << endl;

        Mat hsv;
        cvtColor(img, hsv, COLOR_BGR2HSV);
        string hsv_name = output_folder + "/hsv_test_" + to_string(i + 1) + ".jpg";
        imwrite(hsv_name, hsv);
        cout << "HSV: " << hsv_name << endl;

        auto start = high_resolution_clock::now();

        Mat blob = dnn::blobFromImage(img, 1.0, Size(300, 300),
            Scalar(104, 177, 123), false, false);

        detector.setInput(blob);
        Mat result = detector.forward();

        auto end = high_resolution_clock::now();

        const float* data = result.ptr<float>();
        int faces_found = 0;
        float best_confidence = 0;
        Mat result_img = img.clone();
        vector<Rect> faces;

        for (int j = 0; j < result.size[2]; j++) {
            float confidence = data[j * 7 + 2];

            if (confidence > best_confidence) {
                best_confidence = confidence;
            }

            if (confidence > 0.5) {
                faces_found++;

                int x1 = static_cast<int>(data[j * 7 + 3] * img.cols);
                int y1 = static_cast<int>(data[j * 7 + 4] * img.rows);
                int x2 = static_cast<int>(data[j * 7 + 5] * img.cols);
                int y2 = static_cast<int>(data[j * 7 + 6] * img.rows);
                Rect face_rect(x1, y1, x2 - x1, y2 - y1);
                faces.push_back(face_rect);

                rectangle(result_img, Point(x1, y1), Point(x2, y2),
                    Scalar(0, 255, 0), 2);

                string text = to_string(int(confidence * 100)) + "%";
                putText(result_img, text, Point(x1, y1 - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            }
        }

        string result_name = output_folder + "/result_test_" + to_string(i + 1) + ".jpg";
        imwrite(result_name, result_img);
        cout << "Результат: " << result_name << endl;

        auto time_ms = duration_cast<milliseconds>(end - start);

        cout << "\nОТЧЕТ:" << endl;
        cout << "Найдено лиц: " << faces_found << endl;
        cout << "Статус: ";
        if (faces_found > 0) {
            cout << "ЧЕЛОВЕК ОБНАРУЖЕН" << endl;
            for (int k = 0; k < faces.size(); k++) {
                cout << "  Лицо " << k + 1 << ": " << faces[k].width
                    << "x" << faces[k].height << " пикселей" << endl;
            }
        }
        else {
            cout << "человека нет" << endl;
        }
        cout << "Лучшая уверенность: " << fixed << setprecision(1)
            << best_confidence * 100 << "%" << endl;
        cout << "Время обработки: " << time_ms.count() << " мс" << endl;
    }


    return 0;
}