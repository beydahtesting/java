package com.example.mymcqscannerapp;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import com.github.chrisbanes.photoview.PhotoView;

import org.json.JSONObject;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import java.util.Comparator;
import java.util.List;

public class ResultActivity extends BaseActivity {
    private PhotoView teacherImageView, gradedImageView;
    private TextView resultTextView;
    private Button btnViewResults, btnUploadStudent, btnUploadTeacher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);
        setUpToolbar();
        if(getSupportActionBar() != null){
            getSupportActionBar().setTitle("Results");
        }

        teacherImageView = findViewById(R.id.teacher_result_image_view);
        gradedImageView = findViewById(R.id.graded_image_view);
        resultTextView = findViewById(R.id.result_text_view);
        btnViewResults = findViewById(R.id.btn_view_results);
        btnUploadStudent = findViewById(R.id.btn_upload_student);
        btnUploadTeacher = findViewById(R.id.btn_upload_teacher);

        Bitmap teacherBitmap = ImageCache.getInstance().getTeacherImage();
        Bitmap studentBitmap = ImageCache.getInstance().getStudentImage();

        if (teacherBitmap == null || studentBitmap == null) {
            Toast.makeText(this, "Images missing, please reprocess.", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        teacherImageView.setImageBitmap(teacherBitmap);

        // Convert bitmaps to OpenCV Mats.
        Mat teacherMat = new Mat();
        Mat studentMat = new Mat();
        Utils.bitmapToMat(teacherBitmap, teacherMat);
        Utils.bitmapToMat(studentBitmap, studentMat);

        // Detect circles.
        List<Point> teacherCircles = ImageProcessor.detectFilledCircles(teacherMat);
        List<Point> studentCircles = ImageProcessor.detectFilledCircles(studentMat);

        teacherCircles.sort(Comparator.comparingDouble((Point p) -> p.y)
                .thenComparingDouble(p -> p.x));
        studentCircles.sort(Comparator.comparingDouble((Point p) -> p.y)
                .thenComparingDouble(p -> p.x));

        // Grade: Count correct student answers.
        int correctCount = (int) studentCircles.stream()
                .filter(s -> teacherCircles.stream().anyMatch(t -> Math.hypot(s.x - t.x, s.y - t.y) < 50))
                .count();
        int total = teacherCircles.size();
        String resultText = correctCount + " / " + total + " correct";
        resultTextView.setText(resultText);

        // Draw grading overlay on student image.
        Mat gradedMat = ImageProcessor.compareCircles(teacherCircles, studentCircles, studentMat);
        Imgproc.putText(gradedMat, resultText, new Point(10, 50),
                Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, new org.opencv.core.Scalar(0, 0, 255), 3);
        Bitmap gradedBitmap = Bitmap.createBitmap(gradedMat.cols(), gradedMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(gradedMat, gradedBitmap);
        gradedImageView.setImageBitmap(gradedBitmap);

        // Determine student info based on mode.
        if (AppMode.isOnlineMode()) {
            new Thread(() -> {
                JSONObject apiResponse = ImageProcessor.extractStudentInfo(studentBitmap);
                String candidateText = "";
                try {
                    // Extract candidate text from API response.
                    if (apiResponse.has("candidates")) {
                        candidateText = apiResponse.getJSONArray("candidates")
                                .getJSONObject(0)
                                .getJSONObject("content")
                                .getJSONArray("parts")
                                .getJSONObject(0)
                                .getString("text");
                    }
                } catch (Exception e) {
                    Log.e("GeminiAPI", "Error parsing candidate text", e);
                }
                Log.d("GeminiAPI", "Candidate Text: " + candidateText);

                // Remove asterisks from candidate text if present.
                candidateText = candidateText.replace("*", "");
                String extractedName = ImageProcessor.extractField(candidateText, "Name:");
                String extractedRoll = ImageProcessor.extractField(candidateText, "Roll No.:");
                if (extractedName.isEmpty()) extractedName = "John Doe";
                if (extractedRoll.isEmpty()) extractedRoll = "12345";
                final String finalExtractedName = extractedName;
                final String finalExtractedRoll = extractedRoll;
                runOnUiThread(() -> {
                    StudentRecord.addRecord(new StudentRecord(finalExtractedName, finalExtractedRoll, String.valueOf(correctCount)));
                });
            }).start();
        } else {
            // Offline mode uses dummy data.
            StudentRecord.addRecord(new StudentRecord("John Doe", "12345", String.valueOf(correctCount)));
        }

        // Button handlers.
        btnViewResults.setOnClickListener(v ->
                startActivity(new Intent(ResultActivity.this, StudentResultsActivity.class))
        );

        btnUploadStudent.setOnClickListener(v ->
                startActivity(new Intent(ResultActivity.this, StudentImageActivity.class))
        );

        btnUploadTeacher.setOnClickListener(v ->
                startActivity(new Intent(ResultActivity.this, TeacherImageActivity.class))
        );
    }

    @Override
    protected Intent getForwardIntent() {
        // Forward arrow navigates to StudentResultsActivity.
        return new Intent(ResultActivity.this, StudentResultsActivity.class);
    }
}
