package com.example.mymcqscannerapp

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.net.HttpURLConnection
import java.net.URL
import java.util.Scanner
import java.util.regex.Pattern
import kotlin.math.hypot

object ImageProcessor {
    private val GREEN = Scalar(0.0, 255.0, 0.0)
    private val RED = Scalar(0.0, 0.0, 255.0)
    private const val MATCH_THRESHOLD = 30.0

    // Process image: Ensure correct channel ordering then crop/warp & threshold.
    @JvmStatic
    fun processImage(image: Mat?): Mat? {
        // Check if image is valid.
        var image = image
        if (image == null || image.empty()) {
            Log.e("ImageProcessor", "Input image is null or empty")
            return image
        }

        // Ensure image is in BGR.
        if (image.channels() == 4) {
            val bgr = Mat()
            Imgproc.cvtColor(image, bgr, Imgproc.COLOR_RGBA2BGR)
            image = bgr
        } else if (image.channels() == 1) {
            val bgr = Mat()
            Imgproc.cvtColor(image, bgr, Imgproc.COLOR_GRAY2BGR)
            image = bgr
        }

        // Convert to grayscale.
        val gray = Mat()
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY)

        // Apply Gaussian blur.
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)

        // Apply adaptive thresholding.
        val thresh = Mat()
        Imgproc.adaptiveThreshold(
            blurred, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY_INV, 11, 2.0
        )

        // Edge detection.
        val edges = Mat()
        Imgproc.Canny(thresh, edges, 50.0, 150.0)

        // Find contours.
        val contours: List<MatOfPoint> = ArrayList()
        Imgproc.findContours(
            edges,
            contours,
            Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        // Release temporary Mats.
        gray.release()
        blurred.release()
        thresh.release()
        edges.release()

        if (!contours.isEmpty()) {
            // Find the largest contour.
            var largestContour = contours[0]
            for (cnt in contours) {
                if (Imgproc.contourArea(cnt) > Imgproc.contourArea(largestContour)) {
                    largestContour = cnt
                }
            }

            val perimeter = Imgproc.arcLength(MatOfPoint2f(*largestContour.toArray()), true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(
                MatOfPoint2f(*largestContour.toArray()),
                approx,
                0.02 * perimeter,
                true
            )

            // If a quadrilateral is detected, warp the image.
            if (approx.total() == 4L) {
                val orderedPts = reorderPoints(approx)
                val width = 700.0
                val height = 800.0
                val dst = MatOfPoint2f(
                    Point(0.0, 0.0),
                    Point(width - 1, 0.0),
                    Point(width - 1, height - 1),
                    Point(0.0, height - 1)
                )
                val M = Imgproc.getPerspectiveTransform(orderedPts, dst)
                val warped = Mat()
                Imgproc.warpPerspective(image, warped, M, Size(width, height))

                approx.release()
                orderedPts.release()
                M.release()
                dst.release()

                return warped
            }
        }
        return image.clone()
    }

    // Detect filled circles using an HSV-based blue mask.
    @JvmStatic
    fun detectFilledCircles(image: Mat): List<Point> {
        // Assume the image is in BGR.
        val hsv = Mat()
        Imgproc.cvtColor(image, hsv, Imgproc.COLOR_BGR2HSV)

        val lowerBlue = Scalar(90.0, 50.0, 50.0)
        val upperBlue = Scalar(130.0, 255.0, 255.0)
        val mask = Mat()
        Core.inRange(hsv, lowerBlue, upperBlue, mask)

        val contours: List<MatOfPoint> = ArrayList()
        Imgproc.findContours(
            mask,
            contours,
            Mat(),
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        val filledCircles: MutableList<Point> = ArrayList()
        for (cnt in contours) {
            val area = Imgproc.contourArea(cnt)
            if (area > 100 && area < 5000) {
                val center = Point()
                val radius = FloatArray(1)
                Imgproc.minEnclosingCircle(MatOfPoint2f(*cnt.toArray()), center, radius)
                filledCircles.add(center)
            }
        }
        return filledCircles
    }

    // Compare teacher and student circles:
    // - Correct answers are filled green.
    // - Missing answers are drawn as an empty green circle.
    // - Extra answers are filled in red.
    fun compareCircles(teacherCircles: List<Point>, studentCircles: List<Point>, image: Mat): Mat {
        val correctMatches: MutableList<Point> = ArrayList()
        val unmatchedStudent: List<Point> = ArrayList(studentCircles)
        for (t in teacherCircles) {
            var matchFound = false
            for (s in unmatchedStudent) {
                if (hypot(t.x - s.x, t.y - s.y) < MATCH_THRESHOLD) {
                    correctMatches.add(s)
                    matchFound = true
                    break
                }
            }
            if (!matchFound) {
                // Draw empty green circle for missing answer.
                Imgproc.circle(image, t, 10, GREEN, 1)
            }
        }
        // Draw teacher circles (outline).
        for (t in teacherCircles) {
            Imgproc.circle(image, t, 10, GREEN, 3)
        }
        // Fill correct student answers in green.
        for (s in correctMatches) {
            Imgproc.circle(image, s, 10, GREEN, -1)
        }
        // Mark extra student circles in red.
        for (s in unmatchedStudent) {
            if (!correctMatches.contains(s)) {
                Imgproc.circle(image, s, 10, RED, -1)
            }
        }
        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB)
        return image
    }

    // Draw detected circles for visualization.
    @JvmStatic
    fun drawDetectedCircles(image: Mat): Mat {
        val circles = detectFilledCircles(image)
        val output = image.clone()
        for (p in circles) {
            Imgproc.circle(output, p, 10, GREEN, 2)
        }
        // Convert back to original color space before returning (if necessary)
        return output
    }

    // Warp perspective based on the detected quadrilateral.
    // Reorder points to [top-left, top-right, bottom-right, bottom-left].
    private fun reorderPoints(points: MatOfPoint2f): MatOfPoint2f {
        val pts = points.toArray()
        if (pts.size != 4) return points
        val ordered = arrayOfNulls<Point>(4)
        val sums = DoubleArray(4)
        val diffs = DoubleArray(4)
        for (i in 0..3) {
            sums[i] = pts[i].x + pts[i].y
            diffs[i] = pts[i].y - pts[i].x
        }
        var tl = 0
        var br = 0
        var tr = 0
        var bl = 0
        for (i in 1..3) {
            if (sums[i] < sums[tl]) tl = i
            if (sums[i] > sums[br]) br = i
            if (diffs[i] < diffs[tr]) tr = i
            if (diffs[i] > diffs[bl]) bl = i
        }
        ordered[0] = pts[tl]
        ordered[1] = pts[tr]
        ordered[2] = pts[br]
        ordered[3] = pts[bl]
        return MatOfPoint2f(*ordered)
    }

    // Gemini API call to extract student info.
    // If the API call fails, logs the error and returns a dummy JSONObject.
    @JvmStatic
    fun extractStudentInfo(bitmap: Bitmap): JSONObject {
        try {
            // Compress image and encode as Base64 with NO_WRAP to avoid newlines.
            val imageBytes = ImageUtils.compressToJPEG(bitmap, 30)
            val encodedImage = Base64.encodeToString(imageBytes, Base64.NO_WRAP)

            val inlineData = JSONObject()
            inlineData.put("mime_type", "image/jpeg")
            inlineData.put("data", encodedImage)

            val partsArray = JSONArray()
            partsArray.put(JSONObject().put("inline_data", inlineData))
            partsArray.put(
                JSONObject().put(
                    "text",
                    "Extract only the student's name and roll number from this exam sheet image."
                )
            )

            val contentObject = JSONObject()
            contentObject.put("parts", partsArray)

            val contentsArray = JSONArray()
            contentsArray.put(contentObject)

            val payload = JSONObject()
            payload.put("contents", contentsArray)

            // Log the payload
            Log.d("GeminiAPI", "Payload: $payload")

            val url =
                URL("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyDGalTcZxd_xWk1ZU6SQqgHl3KR5ZvKpoc")
            val conn = url.openConnection() as HttpURLConnection
            conn.doOutput = true
            conn.requestMethod = "POST"
            conn.setRequestProperty("Content-Type", "application/json")

            conn.outputStream.use { os ->
                os.write(payload.toString().toByteArray(charset("UTF-8")))
            }
            val responseCode = conn.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                // If not 200, read the error stream and log it.
                val errorScanner = Scanner(conn.errorStream)
                val errorResponse = if (errorScanner.useDelimiter("\\A")
                        .hasNext()
                ) errorScanner.next() else "No error details"
                errorScanner.close()
                Log.e(
                    "GeminiAPI",
                    "HTTP Error Code: $responseCode Response: $errorResponse"
                )
                throw Exception("HTTP Error Code $responseCode")
            }

            val scanner = Scanner(conn.inputStream)
            val response = if (scanner.useDelimiter("\\A").hasNext()) scanner.next() else ""
            scanner.close()
            Log.d("GeminiAPI", "Response: $response")

            return JSONObject(response)
        } catch (e: Exception) {
            Log.e("GeminiAPI", "Error extracting student info", e)
            // Return dummy values if API call fails.
            try {
                val dummy = JSONObject()
                dummy.put("name", "John Doe")
                dummy.put("rollNumber", "12345")
                return dummy
            } catch (ex: Exception) {
                Log.e("GeminiAPI", "Error creating dummy student info", ex)
                return JSONObject()
            }
        }
    }

    // Helper method: Extract a field value from text using regex.
    // It searches for the given label and returns the text following it until the first newline, quote, or end of string.
    @JvmStatic
    fun extractField(text: String, label: String): String {
        // Replace escaped newlines with actual newlines.
        var text = text
        text = text.replace("\\n", "\n")
        val pattern = Pattern.compile(Pattern.quote(label) + "\\s*(.*?)(\\n|\"|$)")
        val matcher = pattern.matcher(text)
        if (matcher.find()) {
            // Remove asterisks and trim extra whitespace.
            return matcher.group(1).replace("*", "").trim { it <= ' ' }
        }
        return ""
    }
}