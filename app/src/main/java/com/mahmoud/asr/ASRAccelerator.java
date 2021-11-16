package com.mahmoud.asr;

import android.graphics.Bitmap;

import java.util.List;

public class ASRAccelerator implements Classifier {
    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        return null;
    }

    @Override
    public void enableStatLogging(boolean debug) {

    }

    @Override
    public String getStatString() {
        return null;
    }

    @Override
    public void close() {

    }

    @Override
    public void setNumThreads(int num_threads) {

    }

    @Override
    public void setUseNNAPI(boolean isChecked) {

    }

    @Override
    public float getObjThresh() {
        return 0;
    }
}
