package com.example.mymcqscannerapp;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;
import com.github.chrisbanes.photoview.PhotoView;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class TeacherImageActivity extends BaseActivity {
    private static final int REQUEST_CAMERA = 1;
    private static final int REQUEST_GALLERY = 2;
    private Uri photoURI;
    private String currentPhotoPath;
    private Bitmap teacherBitmap;
    private PhotoView imageView;
    private Button btnTeacherCamera, btnTeacherGallery, btnTeacherRotate, btnTeacherFlip, btnTeacherConfirm;
    private TextView tvTeacherLabel;
    private ProgressBar progressBarTeacher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_teacher_image);
        setUpToolbar();

        tvTeacherLabel = findViewById(R.id.tv_teacher_label);
        tvTeacherLabel.setText("Upload Teacher Answer Key");
        if(getSupportActionBar() != null){
            getSupportActionBar().setTitle("Teacher's Key");
        }

        imageView = findViewById(R.id.teacher_image_view);
        btnTeacherCamera = findViewById(R.id.btn_teacher_camera);
        btnTeacherGallery = findViewById(R.id.btn_teacher_gallery);
        btnTeacherRotate = findViewById(R.id.btn_teacher_rotate);
        btnTeacherFlip = findViewById(R.id.btn_teacher_flip);
        btnTeacherConfirm = findViewById(R.id.btn_teacher_confirm);
        progressBarTeacher = findViewById(R.id.progressBarTeacher);
        progressBarTeacher.setVisibility(android.view.View.GONE);

        btnTeacherCamera.setOnClickListener(v -> {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
            } else {
                dispatchTakePictureIntent();
            }
        });

        btnTeacherGallery.setOnClickListener(v -> {
            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_GALLERY);
            } else {
                launchGallery();
            }
        });

        btnTeacherRotate.setOnClickListener(v -> {
            if (teacherBitmap != null) {
                teacherBitmap = rotateImage(teacherBitmap, 90);
                imageView.setImageBitmap(teacherBitmap);
            }
        });

        btnTeacherFlip.setOnClickListener(v -> {
            if (teacherBitmap != null) {
                teacherBitmap = flipImage(teacherBitmap);
                imageView.setImageBitmap(teacherBitmap);
            }
        });

        btnTeacherConfirm.setOnClickListener(v -> {
            if (teacherBitmap != null) {
                ImageCache.getInstance().setTeacherImage(teacherBitmap);
                // Navigate to StudentImageActivity.
                startActivity(new Intent(TeacherImageActivity.this, StudentImageActivity.class));
                finish();
            } else {
                Toast.makeText(this, "Please capture or select a teacher image", Toast.LENGTH_SHORT).show();
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
        String imageFileName = "TEACHER_" + timeStamp + "_";
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
        if ((requestCode == REQUEST_CAMERA && resultCode == RESULT_OK) ||
                (requestCode == REQUEST_GALLERY && resultCode == RESULT_OK && data != null)) {
            progressBarTeacher.setVisibility(android.view.View.VISIBLE);
            new ProcessTeacherImageTask().execute(data);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private class ProcessTeacherImageTask extends AsyncTask<Intent, Void, Bitmap> {
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
                Utils.bitmapToMat(rawBitmap, rawMat);
                Mat processedMat = ImageProcessor.processImage(rawMat);
                // Overlay detected circles on teacher image.
                Mat finalMat = ImageProcessor.drawDetectedCircles(processedMat);
                Bitmap processedBitmap = Bitmap.createBitmap(finalMat.cols(), finalMat.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(finalMat, processedBitmap);
                return processedBitmap;
            } catch (IOException e) {
                e.printStackTrace();
            }
            return null;
        }
        @Override
        protected void onPostExecute(Bitmap result) {
            progressBarTeacher.setVisibility(android.view.View.GONE);
            if (result != null) {
                teacherBitmap = result;
                imageView.setImageBitmap(teacherBitmap);
            } else {
                Toast.makeText(TeacherImageActivity.this, "Error processing teacher image", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // Helper method: rotateImageIfRequired reads EXIF data to determine rotation.
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
        // Forward arrow navigates to StudentImageActivity.
        return new Intent(TeacherImageActivity.this, StudentImageActivity.class);
    }
}
