package com.mahmoud.asr;

import android.content.res.AssetFileDescriptor;
import com.mahmoud.asr.env.Logger;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.bottomsheet.BottomSheetBehavior;
import com.jlibrosa.audio.JLibrosa;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;



public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {
    private MappedByteBuffer tfLiteModel;
    private YoloV5Classifier detector;
    private Interpreter tfLite;
    private Spinner audioClipSpinner;
    private Button transcribeButton;
    private Button playAudioButton;
    private TextView resultTextview;
    private long lastProcessingTimeMs;
    private ImageView plusImageView, minusImageView;
    private TextView inferencetimeinfo;
    private TextView inferencetimeperword;
    private TextView numberofwords;
    private TextView inferencetimepercharacter;
    protected ListView deviceView;
    protected int defaultDeviceIndex = 0;
    protected Handler handler;
    private HandlerThread handlerThread;
    String[] ACC_List;
    private static final Logger LOGGER = new Logger();

    /** Current indices of device and model. */
    int currentDevice = -1;
    int currentModel = -1;
    int currentNumThreads = -1;
    /** holds a gpu delegate */
    GpuDelegate gpuDelegate = null;
    /** holds an nnapi delegate */
    NnApiDelegate nnapiDelegate = null;

    private String wavFilename;
    private MediaPlayer mediaPlayer = new MediaPlayer();
    private final Interpreter.Options tfLiteOptions = new Interpreter.Options();
    private final static String TAG = "TfliteASRDemo";
    private final static int SAMPLE_RATE = 16000 ;
    private final static int DEFAULT_AUDIO_DURATION = -1 ;

    private final static String[] WAV_FILENAMES = {"audio_clip_1.wav", "audio_clip_2.wav", "audio_clip_3.wav"};
    private final static String TFLITE_FILE = "CONFORMER.tflite";
    ArrayList<String> deviceStrings = new ArrayList<String>();
    // protected ArrayList<String> modelStrings = new ArrayList<String>();
    String modelString = "CONFORMER.tflite";

    private LinearLayout bottomSheetLayout;
    protected TextView threadsTextView;
    protected ImageView bottomSheetArrowImageView;
    private LinearLayout gestureLayout;
    private BottomSheetBehavior<LinearLayout> sheetBehavior;

    protected int defaultModelIndex = 0;
    long device_id;

    protected ListView modelView;
    private static final String ASSET_PATH = "";
    JLibrosa jLibrosa = new JLibrosa();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        handler=new Handler();
        audioClipSpinner = findViewById(R.id.audio_clip_spinner);
        ArrayAdapter<String>adapter = new ArrayAdapter<String>(MainActivity.this, android.R.layout.simple_spinner_item, WAV_FILENAMES);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        audioClipSpinner.setAdapter(adapter);
        audioClipSpinner.setOnItemSelectedListener(this);
//PLAY AUDIO
        playAudioButton = findViewById(R.id.play);
        playAudioButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try(AssetFileDescriptor assetFileDescriptor = getAssets().openFd(wavFilename)) {
                    mediaPlayer.reset();
                    mediaPlayer.setDataSource(assetFileDescriptor.getFileDescriptor(), assetFileDescriptor.getStartOffset(), assetFileDescriptor.getLength());
                    mediaPlayer.prepare();
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
                mediaPlayer.start();
            }
        });
        transcribeButton = findViewById(R.id.recognize);
        resultTextview = findViewById(R.id.result);
        inferencetimeinfo = findViewById(R.id.inference_time_info);
        inferencetimeperword = findViewById(R.id.inf_per_word_info);
        numberofwords = findViewById(R.id.number_word_info);
        inferencetimepercharacter = findViewById(R.id.number_char_info);
        Log.v("output1", "0");



        // SET ACC (Device) LIST
        deviceView = findViewById(R.id.device_list);
        deviceStrings.add("CPU");
        deviceStrings.add("GPU");
        deviceStrings.add("NNAPI");
        deviceView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
        ArrayAdapter<String> deviceAdapter =
                new ArrayAdapter<>(
                        this , R.layout.deviceview_row, R.id.deviceview_row_text, deviceStrings);
        deviceView.setAdapter(deviceAdapter);
        deviceView.setItemChecked(defaultDeviceIndex, true);
        final int deviceIndex = deviceView.getCheckedItemPosition();
        deviceView.setOnItemClickListener(
                new AdapterView.OnItemClickListener() {
                    @Override
                    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                        Log.v("output3", "id" + id);

                        device_id = id;

                    }
                });

        // TRANSCRIBTION
        transcribeButton.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View v) {
                if (device_id == 0){
                    useCPU();
                    Log.v("output3", "cpu");
                }else if (device_id == 1){
                    Log.v("output3", "gpu");
                    useGpu();
                }else if(device_id == 2){
                    Log.v("output3", "nnapi");
                    useNNAPI();
                }else {
                    Log.v("output3", "cpu");
                    useCPU();
                }

            }
        });

    }



    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        wavFilename = WAV_FILENAMES[position];
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
        throws IOException{
            AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }

        private String copyWavFileToCache(String wavFilename) {
            File destinationFile = new File(getCacheDir() + wavFilename);
            if (!destinationFile.exists()) {
                try {
                    InputStream inputStream = getAssets().open(wavFilename);
                    int inputStreamSize = inputStream.available();
                    byte[] buffer = new byte[inputStreamSize];
                    inputStream.read(buffer);
                    inputStream.close();

                    FileOutputStream fileOutputStream = new FileOutputStream(destinationFile);
                    fileOutputStream.write(buffer);
                    fileOutputStream.close();
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
            }

            return getCacheDir() + wavFilename;
        }

    private void recreateInterpreter() {
      //  if (tfLite != null) {
          //  tfLite.close();
            try {
                float audioFeatureValues[] = jLibrosa.loadAndRead(copyWavFileToCache(wavFilename), SAMPLE_RATE, DEFAULT_AUDIO_DURATION);
                Object[] inputArray = {audioFeatureValues};
                IntBuffer outputBuffer = IntBuffer.allocate(2000);
                Map<Integer, Object> outputMap = new HashMap<>();
                outputMap.put(0, outputBuffer);
                tfLiteModel = loadModelFile(getAssets(), TFLITE_FILE);
                long startTime = new Date().getTime();
                tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
                tfLite.resizeInput(0, new int[] {audioFeatureValues.length});
                tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
                int outputSize = tfLite.getOutputTensor(0).shape()[0];
                int[] outputArray = new int[outputSize];
                outputBuffer.rewind();
                outputBuffer.get(outputArray);
                StringBuilder finalResult = new StringBuilder();
                for(int i=0; i < outputSize; i++){
                    char c = (char) outputArray[i];
                    if (outputArray[i] != 0){
                        finalResult.append((char) outputArray[i]);
                    }
                }
                lastProcessingTimeMs = new Date().getTime() - startTime;
                String finalResult_str = finalResult.toString();
                String[] splited = finalResult_str.split("\\s+");
                // long totalCharacters = finalResult_str.chars().filter(ch -> ch != ' ').count();
                long totalCharacters = finalResult_str.chars().count();
                int num_of_words = splited.length;
                int inf_per_word = (int) (lastProcessingTimeMs / num_of_words);
                int inf_per_char = (int) (lastProcessingTimeMs / totalCharacters);

                resultTextview.setText(finalResult.toString());
                numberofwords.setText(num_of_words + "  ");
                inferencetimeinfo.setText(lastProcessingTimeMs + "ms");
                inferencetimeperword.setText(inf_per_word + "ms");
                inferencetimepercharacter.setText(inf_per_char + "ms");
            } catch (Exception e){
                Log.e(TAG, e.getMessage());
            }
        }
    public void useCPU() {
        recreateInterpreter();
    }

    public void useGpu() {
        if (gpuDelegate == null) {
            gpuDelegate = new GpuDelegate();
            tfLiteOptions.addDelegate(gpuDelegate);
            recreateInterpreter();
        }

    }
    public void useNNAPI() {
        nnapiDelegate = new NnApiDelegate();
        tfLiteOptions.addDelegate(nnapiDelegate);
        recreateInterpreter();
    }
    }

  //  }

////////////////////////////////////////////////








