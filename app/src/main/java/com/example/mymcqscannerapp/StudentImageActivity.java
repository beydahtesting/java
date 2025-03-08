package com.example.mymcqscannerapp;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.widget.Toast;
import android.widget.Button;
import android.widget.TextView;
import android.widget.ProgressBar;
import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;
import com.github.chrisbanes.photoview.PhotoView;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class StudentImageActivity extends BaseActivity {
    private static final int REQUEST_CAMERA = 1;
    private static final int REQUEST_GALLERY = 2;
    private Uri photoURI;
    private String currentPhotoPath;
    private Bitmap studentBitmap;
    private PhotoView imageView;
    private Button btnCamera, btnGallery, btnRotate, btnFlip, btnStudentConfirm;
    private TextView tvStudentLabel;
    private ProgressBar progressBarStudent;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_student_image);
        setUpToolbar();
        if(getSupportActionBar() != null){
            getSupportActionBar().setTitle("Student's Answer Sheet");
        }

        tvStudentLabel = findViewById(R.id.tv_student_label);
        tvStudentLabel.setText("Upload Student Answer Sheet");

        imageView = findViewById(R.id.student_image_view);
        btnCamera = findViewById(R.id.btn_camera);
        btnGallery = findViewById(R.id.btn_gallery);
        btnRotate = findViewById(R.id.btn_rotate);
        btnFlip = findViewById(R.id.btn_flip);
        btnStudentConfirm = findViewById(R.id.btn_next);
        progressBarStudent = findViewById(R.id.progressBarStudent);
        progressBarStudent.setVisibility(android.view.View.GONE);

        btnCamera.setOnClickListener(v -> dispatchTakePictureIntent());
        btnGallery.setOnClickListener(v -> launchGallery());
        btnRotate.setOnClickListener(v -> {
            if (studentBitmap != null) {
                studentBitmap = rotateImage(studentBitmap, 90);
                imageView.setImageBitmap(studentBitmap);
            }
        });
        btnFlip.setOnClickListener(v -> {
            if (studentBitmap != null) {
                studentBitmap = flipImage(studentBitmap);
                imageView.setImageBitmap(studentBitmap);
            }
        });
        btnStudentConfirm.setOnClickListener(v -> {
            if (studentBitmap != null) {
                ImageCache.getInstance().setStudentImage(studentBitmap);
                // Navigate directly to ResultActivity.
                startActivity(new Intent(StudentImageActivity.this, ResultActivity.class));
                finish();
            } else {
                Toast.makeText(this, "Please capture or select a student image", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void dispatchTakePictureIntent() {
        // Delete previous image file if it exists.
        if (currentPhotoPath != null) {
            File oldFile = new File(currentPhotoPath);
            if (oldFile.exists()) {
                oldFile.delete();
            }
        }

        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            File photoFile;
            try {
                photoFile = createImageFile(); // This method updates currentPhotoPath with the new file path.
            } catch (IOException e) {
                Toast.makeText(this, "Error creating image file", Toast.LENGTH_SHORT).show();
                return;
            }
            photoURI = FileProvider.getUriForFile(this, "com.example.mymcqscannerapp.fileprovider", photoFile);
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
            startActivityForResult(takePictureIntent, REQUEST_CAMERA);
        }
    }


    private File createImageFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "STUDENT_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(imageFileName, ".jpg", storageDir);
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }

    private void launchGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_GALLERY);
    }

    private Bitmap rotateImage(Bitmap img, int degree) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degree);
        return Bitmap.createBitmap(img, 0, 0, img.getWidth(), img.getHeight(), matrix, true);
    }

    private Bitmap flipImage(Bitmap img) {
        Matrix matrix = new Matrix();
        matrix.preScale(-1.0f, 1.0f);
        return Bitmap.createBitmap(img, 0, 0, img.getWidth(), img.getHeight(), matrix, true);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if ((requestCode == REQUEST_CAMERA || requestCode == REQUEST_GALLERY) && resultCode == RESULT_OK) {
            progressBarStudent.setVisibility(android.view.View.VISIBLE);
            new ProcessStudentImageTask().execute(data);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private class ProcessStudentImageTask extends AsyncTask<Intent, Void, Bitmap> {
        @Override
        protected Bitmap doInBackground(Intent... intents) {
            try {
                Bitmap rawBitmap;
                if (intents[0] == null) { // Camera case.
                    File file = new File(currentPhotoPath);
                    rawBitmap = BitmapFactory.decodeFile(file.getAbsolutePath());
                    rawBitmap = rotateImageIfRequired(rawBitmap, photoURI);
                } else { // Gallery case.
                    Uri selectedImage = intents[0].getData();
                    rawBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), selectedImage);
                    rawBitmap = rotateImageIfRequired(rawBitmap, selectedImage);
                }
                Mat rawMat = new Mat();
                org.opencv.android.Utils.bitmapToMat(rawBitmap, rawMat);
                Mat processedMat = ImageProcessor.processImage(rawMat);
                Mat finalMat = ImageProcessor.drawDetectedCircles(processedMat);
                Bitmap processedBitmap = Bitmap.createBitmap(finalMat.cols(), finalMat.rows(), Bitmap.Config.ARGB_8888);
                org.opencv.android.Utils.matToBitmap(finalMat, processedBitmap);
                return processedBitmap;
            } catch (IOException e) {
                e.printStackTrace();
            }
            return null;
        }
        @Override
        protected void onPostExecute(Bitmap result) {
            progressBarStudent.setVisibility(android.view.View.GONE);
            if (result != null) {
                studentBitmap = result;
                imageView.setImageBitmap(studentBitmap);
            } else {
                Toast.makeText(StudentImageActivity.this, "Error processing student image", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private Bitmap rotateImageIfRequired(Bitmap img, Uri imageUri) throws IOException {
        ExifInterface ei = new ExifInterface(getContentResolver().openInputStream(imageUri));
        int orientation = ei.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                return rotateImage(img, 90);
            case ExifInterface.ORIENTATION_ROTATE_180:
                return rotateImage(img, 180);
            case ExifInterface.ORIENTATION_ROTATE_270:
                return rotateImage(img, 270);
            default:
                return img;
        }
    }

    @Override
    protected Intent getForwardIntent() {
        // In StudentImageActivity, the forward arrow navigates to the ResultActivity.
        return new Intent(StudentImageActivity.this, ResultActivity.class);
    }
}
