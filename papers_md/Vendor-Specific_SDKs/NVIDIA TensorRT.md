<!DOCTYPE html>
<html lang='en' class='h-100' data-color-scheme='light'>
  <head>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <meta name="csrf-param" content="authenticity_token" />
<meta name="csrf-token" content="KgWDhckqu5ghJ0YD3uWqipcUQ989zxBzyKvk1qYct923q1VDAj1KkyiyT1HMxPMxMj-iLvwUc59uc536GJE9Ew" />
    <meta name="csp-nonce" />
    <title>TensorRT SDK | NVIDIA Developer</title>
<meta name="description" content="Helps developers to optimize inference, reduce latency, and deliver high throughput for inference applications.">
<meta name="keywords" content="tensorrt, deep learning, inference optimizer, inference platform, sdk, nvidia">
<link rel="canonical" href="https://developer.nvidia.com/tensorrt">
<link rel="alternate" href="https://developer.nvidia.com/tensorrt" hreflang="x-default">
<link rel="alternate" href="https://developer.nvidia.com/tensorrt" hreflang="en-us">
<link rel="alternate" href="https://developer.nvidia.com/ko-kr/tensorrt" hreflang="ko-kr">
<link rel="alternate" href="https://developer.nvidia.cn/tensorrt" hreflang="zh-cn">
<meta property="og:site_name" content="NVIDIA Developer">
<meta property="og:title" content="NVIDIA TensorRT">
<meta property="og:description" content="An SDK with an optimizer for high-performance deep learning inference.">
<meta property="og:type" content="website">
<meta property="og:image" content="https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/tensorrt-getting-started-og-1200x630.jpg">
<meta property="og:url" content="https://developer.nvidia.com/tensorrt">
<meta name="twitter:title" content="NVIDIA TensorRT">
<meta name="twitter:description" content="This SDK enables developers to focus on creating novel AI-powered applications rather than performance tuning for inference deployment.">
<meta name="twitter:image" content="https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/tensorrt-getting-started-og-1200x630.jpg">
<meta name="twitter:site" content="@NVIDIA">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:creator" content="@NVIDIA">
<meta property="interest" content="Developer Tools &amp; Techniques">
      <link rel="alternate" type="text/markdown" title="Page Markdown" href="https://developer.nvidia.com/tensorrt.md">

    <link rel="stylesheet" href="https://developer.nvidia.com/assets/application-f12c0caa0aa49acf1dca3fe463348050c57a8dc40981e85b46b29b0451336be0.css" media="all" />

    <script>
  const NvAssets = {
    CSS: {
      highlightJs: "https://developer.nvidia.com/assets/highlight-7c6157fe69da4cc401cddeb387737d6991b2012dc14e90f3d52764a3a58fd3da.css",
    }
  }
</script>


    
  <link rel="stylesheet" href="https://developer.nvidia.com/assets/one-trust-bea625cf16a072ce5fdb0707a19f2645daf63c05eb1a016db72773eba008fc07.css" />
  <script src="https://cdn.cookielaw.org/scripttemplates/otSDKStub.js" charset="UTF-8" data-document-language="true" data-domain-script="3e2b62ff-7ae7-4ac5-87c8-d5949ecafff5"></script>

<script src="https://images.nvidia.com/aem-dam/Solutions/ot-js/ot-custom.js"></script>

<script>
  function OptanonWrapper() {
    let event = new Event('bannerLoaded');
    window.dispatchEvent(event);

      if (window.OnetrustActiveGroups && window.OnetrustActiveGroups.includes("C0002")) {
        window.DD_RUM && window.DD_RUM.init({
          clientToken: 'pub0430c74fae5d2b467bcb8d48b13e5b32',
          applicationId: '9fc963c7-14e6-403d-bdec-ee671550bb7f',
          site: 'datadoghq.com',
          service: 'devzone',
          env: 'prod',
          version: 'v2.60.0',
          sessionSampleRate: 10,
          sessionReplaySampleRate: 5,
          trackUserInteractions: true,
          trackResources: true,
          trackLongTasks: true,
          defaultPrivacyLevel: 'mask-user-input',
        });
      }
  }
</script>


    <script>
  (function() {
    var didInit = false;
    function initMunchkin() {
      if(didInit === false) {
        didInit = true;
        Munchkin.init('156-OFN-742');
      }
    }
    function loadMunchkin() {
      // Munchkin falls under the OneTrust "Advertising" category (C0004); only
      // load it once the user has consented to that category.
      if (!window.OnetrustActiveGroups || !window.OnetrustActiveGroups.includes("C0004")) {
        return;
      }
      // bannerLoaded re-fires on every consent change; once we have consent we
      // only need to inject the script tag once.
      window.removeEventListener('bannerLoaded', loadMunchkin);
      var s = document.createElement('script');
      s.type = 'text/javascript';
      s.async = true;
      s.src = '//munchkin.marketo.net/munchkin.js';
      s.onreadystatechange = function() {
        if (this.readyState == 'complete' || this.readyState == 'loaded') {
          initMunchkin();
        }
      };
      s.onload = initMunchkin;
      document.getElementsByTagName('head')[0].appendChild(s);
    }
    // OnetrustActiveGroups is populated asynchronously after the OneTrust SDK
    // loads, which dispatches "bannerLoaded" (see layouts/_one_trust). Wait for
    // that signal so consenting users still get Munchkin, then re-check on each
    // consent change (the didInit guard keeps init idempotent).
    window.addEventListener('bannerLoaded', loadMunchkin);
})();
</script>

    <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "Organization",
    "name": "NVIDIA Developer",
    "url": "https://developer.nvidia.com",
    "logo": "https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/_jcr_content/root/responsivegrid/nv_container_392921705/nv_container_412055486/nv_image.coreimg.100.630.png/1703060329095/nvidia-logo-horz.png",
    "sameAs": [
      "https://github.com/nvidia",
      "https://www.linkedin.com/company/nvidia/",
      "https://x.com/nvidiadeveloper"
    ]
  }
</script>

        <meta name='typesense-host' content='typesense.svc.nvidia.com'>
<meta name='typesense-key' content='uFs9XGl9BWS7af7eAIbKNQ49sJnjEfQk'>
<script src="https://developer.download.nvidia.com/scripts/typesense.js"></script>


    <script src="https://assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js" data-ot-ignore="true"></script>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.3/jquery.min.js" integrity="sha512-STof4xm1wgkfm7heWqFJVn58Hm3EtS31XFaagaa8VMReCXAkQnJZ+jEy8PCC/iT18dFy95WcExNHFTqLyp72eQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>

    <script src="https://developer.nvidia.com/assets/bootstrap/5.1.3/bootstrap.bundle.min-51ad1d8cab4ebd9873a0429f5e67ca717a71fd96daf8025bc04a88848e5b375c.js"></script>


    <link rel="icon" type="image/x-icon" href="https://developer.nvidia.com/assets/favicon-81bff16cada05fcff11e5711f7e6212bdc2e0a32ee57cd640a8cf66c87a6cbe6.ico" />
  </head>

  <body class='d-flex flex-column h-100'>
    
    

      <div id='header'></div>




    






    <div id='page-mobile-nav-container'></div>
    <div class='page'>
      <div class="product-page">
<div class="container breadcrumb-container"><div class="col"><ol class="breadcrumb">
<li class="breadcrumb-item"><a href="/topics/" id="ivemf6-2">Topic</a></li>
<div id="ixwtf2-2" class="breadcrumb-item"><a href="/topics/ai/ai-inference/" id="i13tsm">AI Inference</a></div>
<div id="irsh5h-2" class="breadcrumb-item active">TensorRT<br>
</div>
</ol></div></div>
<div class="container page"><div class="row">
<div class="col-xl-2 col-lg-3 col-md-12 col-sm-12 col-sidebar"><aside class="page__sidebar with-sticky-nav"><div class="page-navigation-container"><div data-react-class="PageNavigation" data-react-props='{"draggable":"true","editable":"true","id":"ipvlgw"}' data-react-cache-id="PageNavigation-ipvlgw"></div></div></aside></div>
<div class="col-xl-1 col-lg-1 col-separator with-border"></div>
<div class="col-xl-9 col-lg-9 col-md-12 col-sm-12 col-main-content"><main class="page__content"><section class="page__section page__first-section"><div class="separator separator--no-scale separator--60 d-md-block d-lg-none"></div>
<h1 title="Introduction" class="h--large section__heading">NVIDIA TensorRT</h1>
<p class="p--large text-color-gray mb-0">NVIDIA® TensorRT™ is an ecosystem of tools for developers to achieve high-performance deep learning inference. TensorRT includes inference compilers, runtimes, and model optimizations that deliver low latency and high throughput for production applications. The TensorRT ecosystem includes the TensorRT compiler, TensorRT-LLM, TensorRT Model Optimizer, TensorRT for RTX, and TensorRT Cloud. </p>
<div class="separator separator--45"></div>
<p id="isgvi-3-2"><a href="https://developer.nvidia.com/tensorrt/download" target="" title="" class="btn btn-cta me-2 mt-2">Download Now</a><a href="https://docs.nvidia.com/deeplearning/tensorrt/" target="_blank" title="" class="btn btn-cta--light btn-cta me-2 mt-2">Documentation<br></a><a href="https://github.com/NVIDIA/TensorRT" target="_blank" title="" class="btn btn-cta--light btn-cta me-2 mt-2">GitHub</a></p></section><p class="mb-0"></p>
<section class="page__section page__second-section pb-0 pt-0"> </section><hr class="separator separator--md">
<section class="page__section pt-0 pb-0"><h2 title="How it Works" class="h--medium section__heading toc-item tablet-45">How TensorRT Works</h2>
<p id="iuvaq6" class="p--large">Speed up inference by 36X compared to CPU-only platforms. </p>
<p id="if6mk3">Built on the NVIDIA® CUDA® parallel programming model, TensorRT includes libraries that optimize neural network models trained on all major frameworks, calibrate them for lower precision with high accuracy, and deploy them to hyperscale data centers, workstations, laptops, and edge devices. TensorRT optimizes inference using quantization, layer and tensor fusion, and kernel tuning techniques.<br><br>NVIDIA TensorRT Model Optimizer provides easy-to-use quantization techniques, including post-training quantization and quantization-aware training to compress your models.  FP8, FP4, INT8, INT4, and advanced techniques such as AWQ are supported for your deep learning inference optimization needs. Quantized inference significantly minimizes latency and memory bandwidth, which is required for many real-time services, autonomous and embedded applications.</p>
<img src="https://developer.download.nvidia.com/images/tensorrt/how-tensor-rt-works.jpg" id="ip54jc" class="img-fluid"><div class="separator tablet-45 separator--30"></div>
<div class="row cards__list">
<div id="ic4exo-2-2-3" class="col-sm-12 col-lg-3 col-md-6"><div class="card-wrapper"><div class="card"><div class="card__content">
<h3 class="txt-clr--blck h--smallest">
<span id="docs-internal-guid-1b73ac93-7fff-98aa-bbdc-af9edc339e3a-2-3"><span id="imolkg-2-3">Read the Introductory TensorRT Blog</span></span><br>
</h3>
<p class="mb-0">Learn how to apply TensorRT optimizations and deploy a PyTorch model to GPUs. <br><br></p>
<div class="separator separator--15"></div>
<div class="card__cta"><a href="https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt-updated/" target="_blank" id="iffvbz-4-2-3" class="link-cta text-transform-unset fw-bold">Read Blog</a></div>
</div></div></div></div>
<div id="ic4exo-2-2-3-4" class="col-sm-12 col-lg-3 col-md-6"><div class="card-wrapper"><div class="card"><div class="card__content">
<h3 class="txt-clr--blck h--smallest">
<span id="docs-internal-guid-1b73ac93-7fff-98aa-bbdc-af9edc339e3a-2-3-4"><span id="imolkg-2-3-4">Watch On-Demand TensorRT Sessions From GTC</span></span><br>
</h3>
<p class="mb-0">Learn more about TensorRT and its features from a curated list of webinars at GTC. <br></p>
<div class="separator separator--15"></div>
<div class="card__cta"><a href="https://www.nvidia.com/en-us/on-demand/playlist/playList-53110dbc-c11d-4619-b821-987015090afa/" target="_blank" id="iffvbz-4-2-3-4" class="link-cta text-transform-unset fw-bold">Watch Sessions</a></div>
</div></div></div></div>
<div id="ic4exo-2-2-3-3" class="col-sm-12 col-lg-3 col-md-6"><div class="card-wrapper"><div class="card"><div class="card__content">
<h3 class="txt-clr--blck h--smallest">
<span id="docs-internal-guid-1b73ac93-7fff-98aa-bbdc-af9edc339e3a-2-3-3"><span id="imolkg-2-3-3">Get the Complete Developer Guide</span></span><br>
</h3>
<p class="mb-0">See how to get started with TensorRT in this step-by-step developer and API reference guide.<br><br><br></p>
<div class="separator separator--15"></div>
<div class="card__cta"><a href="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html" target="_blank" id="iffvbz-4-2-3-3" class="link-cta text-transform-unset fw-bold">Read Guide</a></div>
</div></div></div></div>
<div id="ic4exo-2-2-3-2" class="col-sm-12 col-lg-3 col-md-6"><div class="card-wrapper"><div class="card"><div class="card__content">
<h3 class="txt-clr--blck h--smallest">
<span id="docs-internal-guid-1b73ac93-7fff-98aa-bbdc-af9edc339e3a-2-3-2"><span id="imolkg-2-3-2">Navigate AI infrastructure and Performance</span></span><br>
</h3>
<p class="mb-0">Learn how to lower your cost per token and get the most out of your AI models with our ebook.<br><br><br></p>
<div class="separator separator--15"></div>
<div class="card__cta"><a href="https://www.nvidia.com/en-us/solutions/ai/inference/balancing-cost-latency-and-performance-ebook/" target="_blank" id="iffvbz-4-2-3-2" class="link-cta text-transform-unset fw-bold">View Ebook</a></div>
</div></div></div></div>
</div></section><hr class="separator separator--md">
<section class="page__section pt-0 pb-0 section__heading"><h2 title="Key Features" class="h--medium section__heading toc-item tablet-45">Key Features </h2>
<div class="row cards-grid--60">
<div class="col-md-12 col-sm-12 grid-col col-lg-3">
<h3 title="" class="h--smallest">Large Language Model Inference</h3>
<p id="issjxm"><a href="https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/" id="il89ng">NVIDIA TensorRT-LLM</a> is an open-source library that accelerates and optimizes inference performance of large language models (LLMs) on the NVIDIA AI platform with a simplified Python API.<br>Developers accelerate LLM performance on NVIDIA GPUs in the data center or on workstation GPUs. </p>
</div>
<div class="col-md-12 col-sm-12 grid-col col-lg-3">
<h3 title="" class="h--smallest">Compile in the Cloud</h3>
<p id="inv3q8">NVIDIA TensorRT Cloud is a developer-focused service for generating hyper-optimized engines for given constraints and KPIs. Given an LLM and inference throughput/latency requirements, a developer can invoke TensorRT Cloud service using a command-line interface to hyper-optimize a TensorRT-LLM engine for a target GPU. The cloud service will automatically determine the best engine configuration that meets the requirements. Developers can also use the service to build optimized TensorRT engines from ONNX models on a variety of NVIDIA RTX, GeForce, Quadro®, or Tesla®-class GPUs.<br><br>TensorRT Cloud is available with limited access to select partners. <a href="https://developer.nvidia.com/tensorrt-cloud-program" id="i3m9fc" target="_blank">Apply</a> for access, subject to approval.</p>
</div>
<div class="col-md-12 col-sm-12 grid-col col-lg-3">
<h3 title="" class="h--smallest">Optimize Neural Networks</h3>
<p id="ivw8py"><a href="https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/" id="irrvta" target="_blank">TensorRT Model Optimizer</a> is a unified library of state-of-the-art model optimization techniques, including quantization, pruning, speculation, sparsity, and distillation. It compresses deep learning models for downstream deployment frameworks like TensorRT-LLM, TensorRT, vLLM, and SGLang to efficiently optimize inference on NVIDIA GPUs. TensorRT Model Optimizer also supports training for inference techniques such as Speculative Decoding Module Training, Pruning/Distillation, and Quantization Aware Training through NeMo and Hugging Face frameworks.</p>
</div>
<div class="col-md-12 col-sm-12 grid-col col-lg-3">
<h3 title="" class="h--smallest">Major Framework Integrations<br><br>
</h3>
<p id="ikbr3a">TensorRT integrates directly into <a href="https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/" id="ievfvp" target="_blank">PyTorch</a> and <a href="http://hf.co/blog/optimum-nvidia" id="idj5w1" target="_blank">Hugging Face</a> to achieve 6X faster inference with a single line of code. TensorRT provides an <a href="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#fit" id="ixuf7h" target="_blank">ONNX</a> parser to import <a href="https://github.com/NVIDIA/TensorRT/blob/release/10.9/quickstart/IntroNotebooks/2.%20Using%20PyTorch%20through%20ONNX.ipynb" id="ikgtjj" target="_blank">ONNX</a> models from popular frameworks into TensorRT. <a href="https://www.mathworks.com/help/gpucoder/ug/tensorrt-target.html" id="i30akh" target="_blank">MATLAB</a> is integrated with TensorRT through GPU Coder to automatically generate high-performance inference engines for NVIDIA Jetson™, NVIDIA DRIVE®, and data center platforms.</p>
</div>
<div class="col-md-12 col-sm-12 grid-col col-lg-4">
<h3 title="" class="h--smallest">Deploy, Run, and Scale With Dynamo-Triton</h3>
<p id="i8fyem">TensorRT-optimized models are deployed, run, and scaled with <a href="https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/" id="irwv0i">NVIDIA Dynamo Triton</a> inference-serving software that includes TensorRT as a backend. The advantages of using Triton include high throughput with dynamic batching, concurrent model execution, model ensembling, and streaming audio and video inputs.</p>
</div>
<div class="col-md-12 col-sm-12 grid-col col-lg-4">
<h3 title="" class="h--smallest">Simplify AI deployment on RTX</h3>
<p id="i5quxl">TensorRT for RTX offers an optimized inference deployment solution for NVIDIA RTX GPUs. It facilitates faster engine build times within 15 to 30s, facilitating apps to build inference engines directly on target RTX PCs during app installation or on first run, and does so within a total library footprint of under 200 MB, minimizing memory footprint. Engines built with TensorRT for RTX are cross-OS, cross-GPU portable, ensuring a build once, deploy anywhere workflow. <br></p>
</div>
<div class="col-md-12 col-sm-12 grid-col col-lg-4">
<h3 title="" class="h--smallest">Accelerate Every Inference Platform</h3>
<p id="i1zv7l">TensorRT can optimize models for applications across the edge, laptops, desktops, and data centers. It powers key NVIDIA solutions—such as NVIDIA TAO, NVIDIA DRIVE, NVIDIA Clara™, and NVIDIA JetPack™—and is integrated with application-specific SDKs, such as NVIDIA NIM™, NVIDIA DeepStream, NVIDIA Riva, NVIDIA Merlin™, NVIDIA Maxine™, NVIDIA Morpheus, and NVIDIA Broadcast Engine. <br><br>TensorRT provides developers a unified path to deploy intelligent video analytics, speech AI, recommender systems, video conferencing, AI-based cybersecurity, and streaming apps in production.<br></p>
</div>
</div>
<div class="separator tablet-45 separator--30"></div></section><hr class="separator separator--md">
<section class="page__section pt-0 pb-0"><h2 title="Get Started With TensorRT" class="h--medium section__heading toc-item tablet-45">Get Started With TensorRT</h2>
<p class="mb-0">TensorRT is an ecosystem of APIs for building and deploying high-performance deep learning inference. It offers a variety of inference solutions for different developer requirements.</p>
<div class="separator tablet-45 separator--30"></div>
<div class="container"><div id="izc3ll" class="nv-table nv-table-alt-row-2"><table id="iol2o">
<caption class="table-caption"></caption>
<thead class="table-head"><tr>
<th id="i31zd"><p id="immg3y" class="p--large fw-bold">Use-case</p></th>
<th id="i98nn3"><p id="iiznsv" class="p--large fw-bold">Deployment Platform</p></th>
<th id="i2ufo"><p id="iiznsv-2" class="p--large fw-bold">Solution</p></th>
</tr></thead>
<tbody class="table-body">
<tr id="itnvt">
<td id="ioejo"><div id="i1mra9">Inference for LLMs<br><br>
</div></td>
<td data-col="Deployment Platform" id="isvzy"><div id="i1dsbu">Data center GPUs like GB100, H100, A100, etc.</div></td>
<td data-col="Solution" id="i9cemh">
<p id="iiznsv-2-2" class="fw-bold">Download TRT-LLM</p>
<div id="ictavk" class="pb-2">TensorRT-LLM is available for free on <a href="https://github.com/NVIDIA/TensorRT-LLM/tree/rel" id="icse51" target="_blank">GitHub</a>. <br><br>
</div>
<p id="it1mbi-4-3"><a href="https://github.com/NVIDIA/TensorRT-LLM/tree/rel" id="i2fadk-4-3" target="_blank" class="link-cta fw-bold text-transform-unset">Download (GitHub)</a></p>
<p id="itvj4l"><a href="https://nvidia.github.io/TensorRT-LLM" id="i0juup" class="link-cta fw-bold text-transform-unset">Documentation</a></p>
</td>
</tr>
<tr id="ioq9q">
<td id="iy5sc"><div id="iyif5">Inference for non-LLMs like CNNs, Diffusions, Transformers, etc.<br><br>Safety-compliant and high-performance inference for Automotive Embedded<br><br>Inference for non-LLMs in robotics and edge applications</div></td>
<td data-col="Deployment Platform" id="ij0fb"><div id="iuc5ti">Data center GPUs, Embedded, and Edge platforms<br><br><br>Automotive platform: NVIDIA DRIVE AGX<br><br><br>Edge Platform: Jetson, NVIDIA IGX, etc.<br><br>
</div></td>
<td data-col="Solution" id="ishlf">
<p id="i47ngf" class="fw-bold">Download TensorRT</p>
<div id="ictavk-3" class="pb-2">The TensorRT inference library provides a general-purpose AI compiler and an inference runtime that delivers low latency and high throughput for production applications.<br><br>
</div>
<p id="it1mbi"><a href="https://developer.nvidia.com/nvidia-tensorrt-download" id="i2fadk" target="" class="link-cta fw-bold text-transform-unset">Download SDK</a></p>
<p id="it1mbi-2"><a href="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt" id="i2fadk-2" target="_blank" class="link-cta fw-bold text-transform-unset">Download Container</a></p>
</td>
</tr>
<tr id="ioq9q-2">
<td id="iy5sc-2"><div id="iyif5-2">AI Model Inferencing on RTX PCs <br><br>
</div></td>
<td data-col="Deployment Platform" id="ij0fb-2"><div id="iuc5ti-2">NVIDIA GeForce RTX and RTX Pro GPUs in laptops and desktops </div></td>
<td data-col="Solution" id="ishlf-2">
<p id="i47ngf-2" class="fw-bold">Download TensorRT for RTX</p>
<div id="ictavk-3-2" class="pb-2">TensorRT for RTX is a dedicated inference deployment solution for RTX GPUs.<br><br>
</div>
<p id="it1mbi-3"><a href="/tensorrt-rtx" id="i2fadk-3" target="_blank" class="link-cta fw-bold text-transform-unset">Download SDK</a></p>
<p id="it1mbi-3-2"><a href="https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html" id="i2fadk-3-2" target="_blank" class="link-cta fw-bold text-transform-unset">Documentation</a></p>
</td>
</tr>
<tr id="ioq9q-3">
<td id="iy5sc-3"><div id="iyif5-3">Model optimizations like Quantization, Distillation, Sparsity, etc.</div></td>
<td data-col="Deployment Platform" id="ij0fb-3"><div id="iuc5ti-3"><span id="docs-internal-guid-4a82534b-7fff-eb3e-8b51-a6b4649a0fdf">Data center GPUs like GB100, H100, etc.
<br></span></div></td>
<td data-col="Solution" id="ishlf-3">
<p id="i47ngf-3" class="fw-bold">Download TensorRT Model Optimizer</p>
<div id="ictavk-3-3" class="pb-2">TensorRT Model Optimizer is free on NVIDIA PyPI, with examples and recipes on <a href="https://github.com/NVIDIA/TensorRT-Model-Optimizer" id="ixxzvu">GitHub</a>. <br><br>
</div>
<p id="it1mbi-4"><a href="https://github.com/NVIDIA/TensorRT-Model-Optimizer" id="i2fadk-4" target="_blank" class="link-cta fw-bold text-transform-unset">Download (GitHub)</a></p>
<p id="it1mbi-2-3"><a href="https://nvidia.github.io/TensorRT-Model-Optimizer" id="i2fadk-2-3" target="_blank" class="link-cta fw-bold text-transform-unset">Documentation</a></p>
</td>
</tr>
</tbody>
</table></div></div></section><hr class="separator separator--md">
<section class="page__section pt-0 pb-0"><h2 title="Get Started With Torch-TensorRT" class="h--medium section__heading toc-item tablet-45">Get Started With TensorRT Frameworks<br>
</h2>
<p class="mb-0">TensorRT Frameworks add TensorRT compiler functionality to frameworks like PyTorch.</p>
<div class="separator tablet-45 separator--30"></div>
<div id="i8d82z-4"><div class="row cards__list">
<div class="col-sm-12 grid-col col-md-6 col-lg-6 col-xl-4">
<img src="https://developer.download.nvidia.com/icons/m48-download.svg" id="ivqf9k-2-5" alt="TensorRT speeds up inference by 36X" class="img-fluid mw-6-rem"><h3 class="h--smaller mb-0">Download ONNX and Torch-TensorRT</h3>
<div class="separator separator--15"></div>
<p class="mb-0">The TensorRT inference library provides a general-purpose AI compiler and an inference runtime that delivers low latency and high throughput for production applications.</p>
<div class="separator tablet-45 separator--30"></div>
<p class="mb-0"><b id="inybk5" class="h--smallest">ONYX:</b></p>
<div id="ickxpc-2"><a id="ibi58v-3" href="https://github.com/NVIDIA/TensorRT/blob/release/10.9/quickstart/IntroNotebooks/2.%20Using%20PyTorch%20through%20ONNX.ipynb" target="_blank" class="link-cta text-transform-unset fw-bold gjs-selected">Documentation</a></div>
<div class="separator tablet-45 separator--30"></div>
<p class="mb-0"><b id="i7gari" class="h--smallest">Torch-TensorRT:</b></p>
<div id="ickxpc-2-3"><a id="ibi58v-3-3" href="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch" target="_blank" class="link-cta text-transform-unset fw-bold gjs-selected">Download Container</a></div>
<div id="ickxpc-2-2"><a id="ibi58v-3-2" href="https://pytorch.org/TensorRT/" target="_blank" class="link-cta text-transform-unset fw-bold gjs-selected">Documentation</a></div>
<div class="separator separator--15"></div>
</div>
<div class="col-sm-12 grid-col col-md-6 col-lg-6 col-xl-4">
<img src="https://developer.download.nvidia.com/icons/m48-accellerated-computing-with-cuda-python-256px-blk.png" id="ivqf9k-2-2-2" alt="TensorRT speeds up inference by 36X" class="img-fluid mw-6-rem"><h3 class="h--smaller mb-0">Experience Tripy: Pythonic Inference With TensorRT</h3>
<div class="separator separator--15"></div>
<p class="mb-0">Experience high-performance inference and excellent usability with Tripy. Expect intuitive APIs, easy debugging with eager mode, clear error messages, and top-notch documentation to streamline your deep learning deployment.</p>
<div class="separator tablet-45 separator--30"></div>
<div id="ickxpc-2-3-4"><a id="ibi58v-3-3-4" href="https://nvidia.github.io/TensorRT-Incubator/index.html" target="_blank" class="link-cta text-transform-unset fw-bold gjs-selected">Documentation</a></div>
<div id="ickxpc-2-3-3"><a id="ibi58v-3-3-3" href="https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy/examples" target="_blank" class="link-cta text-transform-unset fw-bold gjs-selected">Examples</a></div>
<div id="ickxpc-2-3-2"><a id="ibi58v-3-3-2" href="https://github.com/NVIDIA/TensorRT-Incubator/blob/main/tripy/CONTRIBUTING.md" target="_blank" class="link-cta text-transform-unset fw-bold gjs-selected">Contribute</a></div>
<div class="separator separator--15"></div>
</div>
<div class="col-sm-12 grid-col col-md-6 col-lg-6 col-xl-4">
<img src="https://developer.download.nvidia.com/icons/m48-digital-deep-learning-institute-talks-training.svg" id="ivqf9k-2-4-2" alt="TensorRT speeds up inference by 36X" class="img-fluid mw-6-rem"><h3 class="h--smaller mb-0">Deploy</h3>
<div class="separator separator--15"></div>
<p class="mb-0">Get a free license to try <a href="https://www.nvidia.com/en-us/data-center/products/ai-enterprise/" id="igcpob" target="_blank">NVIDIA AI Enterprise</a> in production for 90 days using your existing infrastructure.</p>
<div class="separator tablet-45 separator--30"></div>
<div id="ickxpc-2-3-4-2"><a id="ibi58v-3-3-4-2" href="https://enterpriseproductregistration.nvidia.com/?LicType=EVAL&amp;ProductFamily=NVAIEnterprise" target="_blank" class="link-cta text-transform-unset fw-bold gjs-selected">Request a 90-Day License</a></div>
<div class="separator separator--15"></div>
</div>
</div></div>
<div class="row cards-grid--60"></div></section><hr class="separator separator--md">
<section class="page__section page__second-section pb-0 pt-0"><h2 title="Performance" class="h--medium section__heading toc-item">World-Leading Inference Performance<br>
</h2>
<p id="i036y2-2">TensorRT was behind NVIDIA’s wins across all <a href="https://developer.nvidia.com/blog/tag/inference-performance/" id="imckv6">inference performance</a> tests in the industry-standard benchmark for <a href="https://www.nvidia.com/en-us/data-center/mlperf/" id="ivdyvw">MLPerf Inference</a>. TensorRT-LLM accelerates the latest large language models for <a href="https://www.nvidia.com/en-us/ai-data-science/generative-ai/" id="i7al0d">generative AI</a>, delivering up to 8X more performance, 5.3X better TCO, and nearly 6X lower energy consumption.</p>
<a id="iv1huc-2" href="/deep-learning-performance-training-inference/ai-inference" target="" class="link-cta text-transform-unset fw-bold">See All Benchmarks</a><div class="separator separator--15"></div>
<div class="row cards-grid--60">
<div class="col-sm-12 grid-col col-md-6">
<h3 class="h--smaller mb-0 text-center">8X Increase in GPT-J 6B Inference Performance</h3>
<div class="separator separator--15"></div>
<img src="https://developer.download.nvidia.com/images/gpt-j-6b-630x354-1.jpg" alt="TensorRT-LLM on H100 has 8X increase in GPT-J 6B inference performance" id="iwv0cg-2" class="img-fluid">
</div>
<div class="col-sm-12 grid-col col-md-6">
<h3 class="h--smaller mb-0 text-center">4X Higher Llama2 Inference Performance<br>
</h3>
<div class="separator separator--15"></div>
<img src="https://developer.download.nvidia.com/images/llama-2-70b-630x354-1.jpg" alt="TensorRT-LLM on H100 has 4X Higher Llama2 Inference Performance" id="itkmqd-2" class="img-fluid">
</div>
<div class="col-sm-12 grid-col col-md-6">
<h3 class="h--smaller mb-0 text-center">Total Cost of Ownership</h3>
<div id="idrtzf-2" class="text-center">Lower is better</div>
<div class="separator separator--15"></div>
<img src="https://developer.download.nvidia.com/images/cost-of-ownership-630x354-1.jpg" alt="TensorRT-LLM has lower total cost of ownership than GPT-J 6B and Llama 2 70B" id="imyxg1-2" class="img-fluid">
</div>
<div class="col-sm-12 grid-col col-md-6">
<h3 class="h--smaller mb-0 text-center">Energy Use</h3>
<div id="i1p9wl-2" class="text-center">Lower is better</div>
<div class="separator separator--15"></div>
<img src="https://developer.download.nvidia.com/images/energy-use-630x354-1.jpg" alt="TensorRT-LLM has lower energy use than GPT-J 6B and Llama 2 70B" id="iix4it-2" class="img-fluid">
</div>
</div></section><hr class="separator separator--md">
<section class="page__section page__second-section pt-0 pb-0"><div class="row nv-waterfall">
<div class="col-md-12 col-sm-12 text-column js-text-column col-lg-6 col-xl-6">
<h4 class="h--small">NVIDIA Blackwell Ultra Delivers up to 50x Better Performance and 35x Lower Cost for Agentic AI</h4>
<p id="iy33ph">Built to accelerate the next generation of agentic AI, NVIDIA Blackwell Ultra delivers breakthrough inference performance with dramatically lower cost. Cloud providers such as Microsoft, CoreWeave, and Oracle Cloud Infrastructure are deploying NVIDIA GB300 NVL72 systems at scale for low-latency and long-context use cases, such as agentic coding and coding assistants.<br><br>This is enabled by deep co-design across NVIDIA Blackwell, NVLink™, and NVLink Switch for scale-out; NVFP4 for low-precision accuracy; and NVIDIA Dynamo and TensorRT™ LLM for speed and flexibility—as well as development with community frameworks SGLang, vLLM, and more.<br></p>
<div class="card__cta"><a href="https://developer.nvidia.com/deep-learning-performance-training-inference/ai-inference" target="" id="iffvbz-4-2-3-5-4-2" class="link-cta text-transform-unset fw-bold">Explore technical results</a></div>
</div>
<div class="col-md-12 col-sm-12 image-column js-image-column col-lg-6 col-xl-6"><figure id="iqvxook" title="Data center illustration showing multi-modal AI tokens for image, audio, visual and more as part of the NVIDIA “Think SMART” framework."><img src="https://developer.download.nvidia.com/images/dgx-press-gb300-1920x1080.jpg" alt="Data center illustration showing multi-modal AI tokens for image, audio, visual and more as part of the NVIDIA “Think SMART” framework." class="card-img-top"></figure></div>
</div></section><hr class="separator separator--md">
<section class="page__section pt-0 pb-0"><h2 id="i31rjh-2" title="Starter Kits" class="h--medium toc-item section__heading">Starter Kits</h2>
<div class="row cards-grid--60">
<div class="col-md-12 col-sm-12 grid-col col-lg-4">
<h3 title="" class="mb-0 h--smallest">Beginner Guide to TensorRT<br>
</h3>
<div class="separator separator--30"></div>
<ul id="i4agjt-2" class="nv-list">
<li id="ios0ei-2" data-icon="file"><div id="i08hlv-2"><p class="mb-0"><a href="/tensorrt-getting-started" id="ie4kdc-2" target="">View Quick-Start Guide</a></p></div></li>
<li id="i5xs6i-2" data-icon="file"><div id="ike3qd-2"><p class="mb-0"><a href="https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html" id="izf9sg-2" target="_blank">View Quick-Start Notebooks</a></p></div></li>
<li id="ipo9zk-3" data-icon="file"><div id="id2umv-3">
<p class="mb-0"></p>
<p id="idjoql5">Read Blog: <a href="/blog/speeding-up-deep-learning-inference-using-tensorrt-updated/" id="ip8crq6" target="">Speeding Up Deep Learning Inference Using NVIDIA TensorRT</a></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-2" data-icon="file"><div id="id2umv-3-2">
<p class="mb-0"></p>
<p id="idjoql5-2">Read Blog: <a href="/blog/optimizing-and-serving-models-with-nvidia-tensorrt-and-nvidia-triton/" id="ip8crq6-2" target="">Optimizing and Serving Models With TensorRT and Triton</a></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-2-2" data-icon="play"><div id="id2umv-3-2-2">
<p class="mb-0"></p>
<p id="idjoql5-2-2">Watch Video: <a href="https://www.youtube.com/watch?v=SlUouzxBldU" id="if4j4hg">Getting Started With NVIDIA TensorRT</a></p>
<br><p></p>
</div></li>
</ul>
<div class="separator separator--30"></div>
</div>
<div class="col-md-12 col-sm-12 grid-col col-lg-4">
<h3 id="iwqubu3" title="" class="mb-0 h--smallest gjs-selected">Beginner Guide to TensorRT-LLM</h3>
<div class="separator separator--30"></div>
<ul id="i4agjt-2-2" class="nv-list">
<li id="ios0ei-2-2" data-icon="file"><div id="i08hlv-2-2"><p class="mb-0"><a href="/tensorrt-getting-started" id="ie4kdc-2-2" target="">View Quick-Start Guide</a></p></div></li>
<li id="i5xs6i-2-2" data-icon="file"><div id="ike3qd-2-2"><p class="mb-0"><a href="https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html" id="izf9sg-2-2" target="_blank">View Quick-Start Notebooks</a></p></div></li>
<li id="ipo9zk-3-3" data-icon="file"><div id="id2umv-3-3">
<p class="mb-0"></p>
<p id="idjoql5-3">Read Blog: <a href="/blog/speeding-up-deep-learning-inference-using-tensorrt-updated/" id="ip8crq6-3" target="_blank">Speeding Up Deep Learning Inference Using NVIDIA TensorRT</a></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-2-3" data-icon="file"><div id="id2umv-3-2-3">
<p class="mb-0"></p>
<p id="idjoql5-2-3">Read Blog: <a href="/blog/optimizing-and-serving-models-with-nvidia-tensorrt-and-nvidia-triton/" id="ip8crq6-2-2" target="">Optimizing and Serving Models With TensorRT and Triton</a></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-2-2-2" data-icon="file"><div id="id2umv-3-2-2-2">
<p class="mb-0"></p>
<p id="idjoql5-2-2-2">Watch Video: <a href="https://www.youtube.com/watch?v=SlUouzxBldU" id="if4j4hg-2">Getting Started With NVIDIA TensorRT</a></p>
<br><p></p>
</div></li>
</ul>
<div class="separator separator--30"></div>
</div>
<div class="col-md-12 col-sm-12 grid-col col-lg-4">
<h3 title="" class="mb-0 h--smallest">Beginner Guide to TensorRT Model Optimizer<br>
</h3>
<div class="separator separator--30"></div>
<ul class="nv-list">
<li data-icon="file" id="i0gulg-3"><div><p class="mb-0"><a href="https://docs.omniverse.nvidia.com/simready/latest/sim-needs/synth-data-gen.html" target="_blank" id="i642us-3">Reference Architecture</a></p></div></li>
<li id="ipo9zk-2-5-2" data-icon="file"><div id="id2umv-2-5-2"><p class="mb-0"><a href="https://docs.omniverse.nvidia.com/extensions/latest/ext_product-configurator.html" id="ivwz96-2-5-2" target="_blank">Workflow Guide &amp; Documentation</a><br></p></div></li>
<li id="ipo9zk-2-4-2" data-icon="file"><div id="id2umv-2-4-2"><p class="mb-0"><a href="https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-OV-14+V1" id="ivwz96-2-4-2" target="_blank">Training Courses</a><br></p></div></li>
<li id="ipo9zk-2-3-2" data-icon="file"><div id="id2umv-2-3-2"><p class="mb-0"><a href="https://build.nvidia.com/nvidia/conditioning-for-precise-visual-generative-ai" id="ivwz96-2-3-2" target="_blank">NVIDIA Omniverse Blueprint for Precise Visual Generative AI</a><br></p></div></li>
</ul>
<div class="separator separator--30"></div>
</div>
<div class="col-md-12 col-sm-12 grid-col col-lg-4">
<h3 title="" class="mb-0 h--smallest">Beginner Guide to Torch-TensorRT<br>
</h3>
<ul class="nv-list">
<li id="ipo9zk-3-6" data-icon="play"><div id="id2umv-3-6">
<p class="mb-0"></p>
<p id="idjoql5-6">Watch Video: <a href="https://www.youtube.com/watch?v=TU5BMU6iYZ0" id="ip8crq6-5" target="_blank">Getting Started With NVIDIA Torch-TensorRT</a></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-4" data-icon="file"><div id="id2umv-3-4">
<p class="mb-0"></p>
<p id="idjoql5-4">Read Blog: <a href="/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/" id="ip8crq6-6" target="">Accelerate Inference up to 6X in PyTorch</a></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-4-3" data-icon="external-link"><div id="id2umv-3-4-3">
<p class="mb-0"></p>
<p id="idjoql5-4-3">Download Notebook: <a href="https://github.com/NVIDIA/Torch-TensorRT/blob/master/notebooks/ssd-object-detection-demo.ipynb" id="iia645" target="_blank">Object Detection With SSD</a> (Jupyter Notebook)</p>
<br><p></p>
</div></li>
</ul>
<div class="separator separator--30"></div>
</div>
<div class="col-md-12 col-sm-12 grid-col col-lg-4">
<h3 title="" class="mb-0 h--smallest">Beginner Guide to TensorRT Pythonic Frontend: Tripy<br>
</h3>
<ul class="nv-list">
<li id="ipo9zk-3-4-2" data-icon="file"><div id="id2umv-3-4-2">
<p class="mb-0"></p>
<p id="idjoql5-4-2"><a href="https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/00-introduction-to-tripy.html" id="ioeq7m" target="_blank">Introduction Guide</a></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-4-2-4" data-icon="file"><div id="id2umv-3-4-2-4">
<p class="mb-0"></p>
<p id="idjoql5-4-2-4"><a href="https://github.com/NVIDIA/TensorRT-Incubator/blob/main/tripy/notebooks/resnet50.ipynb" id="ioeq7m-4" target="_blank">ResNet-50 notebook</a></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-4-2-3" data-icon="file"><div id="id2umv-3-4-2-3">
<p class="mb-0"></p>
<p id="idjoql5-4-2-3"><a href="https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy/examples/nanogpt" id="ioeq7m-3" target="_blank">nanoGPT</a></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-4-2-2" data-icon="file"><div id="id2umv-3-4-2-2">
<p class="mb-0"></p>
<p id="idjoql5-4-2-2"><a href="https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy/examples/segment-anything-model-v2" id="ioeq7m-2" target="_blank">Segment Anything Model V2</a></p>
<br><p></p>
</div></li>
</ul>
<div class="separator separator--30"></div>
</div>
<div class="col-md-12 col-sm-12 grid-col col-lg-4">
<h3 title="" class="mb-0 h--smallest">Beginner Guide to TensorRT for RTX<br>
</h3>
<ul class="nv-list">
<li id="ipo9zk-3-4-2-5" data-icon="file"><div id="id2umv-3-4-2-5">
<p class="mb-0"></p>
<p id="idjoql5-4-2-5"><a href="https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/installing.html" id="ioeq7m-5" target="_blank">View Quick Start Guide<br></a></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-4-2-4-2" data-icon="file"><div id="id2umv-3-4-2-4-2">
<p class="mb-0"></p>
<p id="idjoql5-4-2-4-2"><a href="https://github.com/NVIDIA/TensorRT-RTX/tree/main" id="ioeq7m-4-2" target="_blank">Access Samples and Demos</a></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-4-2-3-2" data-icon="file"><div id="id2umv-3-4-2-3-2">
<p class="mb-0"></p>
<p id="idjoql5-4-2-3-2"><a href="https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-for-rtx/" id="ioeq7m-3-2" target="_blank">Read Blog: </a></p>
<h1 id="iitubb" class="h--large txt-clr--blck mt-2 mb-0"><a href="https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-for-rtx/" id="ioeq7m-3-2" target="_blank">Run High-Performance AI Applications with NVIDIA TensorRT for RTX</a></h1>
<a href="https://developer.nvidia.com/blog/run-high-performance-ai-applications-with-nvidia-tensorrt-for-rtx/" id="ioeq7m-3-2" target="_blank"> <br></a><p></p>
<br><p></p>
</div></li>
<li id="ipo9zk-3-4-2-3-2-2" data-icon="file"><div id="id2umv-3-4-2-3-2-8">
<p class="mb-0"></p>
<p id="idjoql5-4-2-3-2-8"><a href="https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/get-started?tabs=csharp" id="ioeq7m-3-2-8" target="_blank">Access TensorRT for RTX through WindowsML <br></a></p>
<br><p></p>
</div></li>
</ul>
<div class="separator separator--30"></div>
</div>
</div></section><hr class="separator separator--md">
<section class="page__section pt-0 pb-0"><h2 title="Learning Library" class="h--medium section__heading toc-item">TensorRT Learning Library</h2>
<div id="ixdxnl"><div class="row cards__list">
<div class="col-xl-4 col-lg-4 col-md-12 col-sm-12"><div class="card-wrapper"><div class="card"><div class="card__content"><div id="iauods">
<div class="separator separator--15"></div>
        <div class="custom-html-wrapper">
          <div class="custom-html-wrapper__code"><div class="card-badges">
  <div class="badge badge--code">OSS (Github)
  </div>
</div></div>
          <div data-react-class="CustomHtml" data-react-props='{"preview":false}'></div>
        </div>
        <p id="ip29ok-2-2-2-3-2-2" class="h--smallest text-start">Quantization Quickstart</p>
<div class="separator separator--30"></div>
<p class="mb-0"><b>NVIDIA TensorRT-LLM</b><br><br>The <a href="https://nvidia.github.io/TensorRT-LLM/torch.html#quantization" id="iuv3wz">PyTorch backend</a> supports FP8 and NVFP4 quantization. Explore <a href="https://nvidia.github.io/TensorRT-LLM/torch.html#quantization" id="igpjgn">GitHub</a> to pass quantized models in the Hugging Face model hub, which are generated by TensorRT Model Optimizer.<br>  <br><a href="https://nvidia.github.io/TensorRT-LLM/torch.html#quantization" id="ib4mqi" target="_blank">Link to GitHub</a><br><a href="https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html" id="ie84ey" target="_blank">Link to PyTorch Documentation</a><br></p>
</div></div></div></div></div>
<div class="col-xl-4 col-lg-4 col-md-12 col-sm-12"><div class="card-wrapper"><div class="card"><div class="card__content"><div id="iauods-3">
<div class="separator separator--15"></div>
        <div class="custom-html-wrapper">
          <div class="custom-html-wrapper__code"><div class="card-badges">
  <div class="badge badge--code">OSS (Github)
  </div>
</div></div>
          <div data-react-class="CustomHtml" data-react-props='{"preview":false}'></div>
        </div>
        <p id="ip29ok-2-2-2-3-2-2-3" class="h--smallest text-start">Adding a New Model in PyTorch Backend</p>
<div class="separator separator--30"></div>
<p class="mb-0">This guide provides a step-by-step process for adding a new model in PyTorch Backend.<br><br><a href="https://nvidia.github.io/TensorRT-LLM/torch/adding_new_model.html" id="imcy3m" target="_blank">Link to GitHub</a></p>
</div></div></div></div></div>
<div class="col-xl-4 col-lg-4 col-md-12 col-sm-12"><div class="card-wrapper"><div class="card"><div class="card__content"><div id="iauods-2">
<div class="separator separator--15"></div>
        <div class="custom-html-wrapper">
          <div class="custom-html-wrapper__code"><div class="card-badges">
  <div class="badge badge--code">OSS (Github)
  </div>
</div></div>
          <div data-react-class="CustomHtml" data-react-props='{"preview":false}'></div>
        </div>
        <p id="ip29ok-2-2-2-3-2-2-2" class="h--smallest text-start">Using TensoRT-Model Optimizer for Speculative Decoding<br></p>
<div class="separator separator--30"></div>
<p class="mb-0">ModelOpt’s Speculative Decoding module enables your model to generate multiple tokens in each generation step. This can be useful for reducing the latency of your model and speeding up inference.<br><br><a href="https://nvidia.github.io/TensorRT-Model-Optimizer/guides/7_speculative_decoding.html" id="i8y5nh" target="_blank">Link to GitHub</a></p>
</div></div></div></div></div>
</div></div></section><hr class="separator separator--md">
<section class="page__section"><h2 title="Ecosystem" class="h--medium section__heading toc-item">TensorRT Ecosystem Ecosystem</h2>
<p id="i10hkx-3-2" class="p--large">Widely Adopted Across Industries</p>
<div class="separator separator--30"></div>
<div id="iwzaux-3">
<div class="row"><div class="col-xs-12 col-lg-12"><img src="https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/tensorrt/Logo_farm_GTC.png" id="ig71t" alt="NVIDIA TensorRT is widely adopted by top companies across industries"></div></div>
<div id="iwzaux-2-2"></div>
</div></section><hr class="separator separator--md">
<section class="page__section page__last-section pb-0 pt-0"><h2 title="More Resources" class="h--medium section__heading toc-item tablet-45">More Resources</h2>
<div id="iedinv-2-3"><div class="row cards__list">
<div id="iucjy8-2-3" class="col-lg-4 col-md-4 col-sm-12"><div class="card-wrapper">
<a href="https://developer.nvidia.com/email-signup" id="ib782k-2-3" target=""></a><div class="card extra-resource">
<img alt="NVIDIA Developer Forums" src="https://developer.download.nvidia.com/icons/m48-people-group.svg" class="img-fluid mw-6-rem mx-auto"><a href="https://developer.download.nvidia.com/icons/m48-people-group.svg" target="" id="iq4jgq-2-3"></a><h3 class="txt-clr--blck mb-0 text-center h--smallest">Explore the Community</h3>
</div>
</div></div>
<div id="ibgu4h-2-3" class="col-lg-4 col-md-4 col-sm-12"><div class="card-wrapper">
<a href="https://www.nvidia.com/en-us/training/" id="ibllhc-2-3" target="_blank"></a><div class="card extra-resource">
<img src="https://developer.download.nvidia.com/icons/m48-certification-ribbon-2.svg" id="ifszp9-2-3" alt="NVIDIA Training and Certification" class="img-fluid mw-6-rem mx-auto"><h3 class="txt-clr--blck mb-0 text-center h--smallest">Get Training and Certification<br>
</h3>
<a href="https://www.nvidia.com/en-us/training/" target="_blank" id="i462f2-2-3"></a>
</div>
</div></div>
<div id="ir7h78-2-3" class="col-lg-4 col-md-4 col-sm-12"><div class="card-wrapper">
<a href="https://resources.nvidia.com/en-us-inference-perf-q1-fy26-nurture" id="i7lwq8-2-3" target="_blank"></a><div class="card extra-resource">
<img src="https://developer.download.nvidia.com/images/isaac/m48-ai-startup-256px-blk.png" id="i9pjs2-2-3" alt="NVIDIA Inception Program for Startups" class="img-fluid mw-6-rem mx-auto"><h3 class="txt-clr--blck mb-0 text-center h--smallest">Read Top Stories and Blogs</h3>
<a href="https://resources.nvidia.com/en-us-inference-perf-q1-fy26-nurture" target="" id="i16c9u-2-3"></a>
</div>
</div></div>
</div></div>
<ul class="nv-list"></ul></section><hr class="separator separator--md">
<section class="page__section pt-0"><h2 class="h--medium section__heading">Ethical AI</h2>
<p id="ids1ek">NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.<br><br>For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety &amp; Security, and Privacy Subcards. Please report security vulnerabilities or NVIDIA AI Concerns <a href="https://www.nvidia.com/en-us/support/submit-security-vulnerability/" id="ikhhjp3" target="_blank">here</a><a href="https://www.nvidia.com/en-us/support/submit-security-vulnerability/" id="i0p1en" target="_blank"></a>.</p></section><section class="page__section page__section--light-gray page__last-section page__cta-section section--gray"><p id="i6mxj2" class="p--large text-center"><b id="il0r2k">Get started with TensorRT today, and use the right inference tools to develop AI for any application on any platform.</b></p>
<div class="separator separator--30"></div>
<p class="text-center mb-0"><a href="https://developer.nvidia.com/tensorrt/download" target="" class="btn btn-cta">Download Now<br></a></p></section></main></div>
</div></div>
<div class="separator separator--90 phone-0"></div>
</div>
    </div>






          <script type="text/javascript">
  (() => {
    const handleQuotesBlock = (quotesBlock, idx) => {
      const blockquotes = quotesBlock.querySelectorAll('blockquote');
      if (blockquotes.length < 1) {
        return;
      }

      const navContainer = document.createElement('ul');
      navContainer.classList.add('quotes-list-navigation');
      for (let i = 0; i < blockquotes.length; i++) {
        let navItem = document.createElement('li');
        let btn = document.createElement('button');
        btn.type = 'button';
        btn.dataset['group'] = idx.toString();
        btn.dataset['length'] = blockquotes.length.toString();
        btn.value = i.toString();

        btn.addEventListener('click', (e) => {
          const group = e.target.dataset['group'];
          const groupActiveButtons = document.querySelectorAll(`button[data-group="${group}"].active`);
          groupActiveButtons.forEach((activeButton) => {
            activeButton.classList.remove('active');
          });
          e.target.classList.add('active');
          const viewPortWidth = quotesBlock.getBoundingClientRect().width;
          const clickedSlide = parseInt(e.target.value);
          quotesBlock.querySelector('.quotes-list').style.transform = `translate(-${viewPortWidth * clickedSlide}px)`;
        });
        navItem.appendChild(btn);
        navContainer.appendChild(navItem);

        if (i === 0) {
          btn.click();
        }
      }
      quotesBlock.appendChild(navContainer);
    };

    const refreshQuotesBlock = () => {
      document.querySelectorAll('.quotes-list-navigation button.active').forEach((b) => {
        const currentItem = parseInt(b.value);
        const maxItem = parseInt(b.dataset['length']);
        const group = parseInt(b.dataset['group']);
        const next = currentItem + 1;
        if (next < maxItem) {
          document.querySelectorAll(`button[data-group="${group}"]`)[next].click();
        } else {
          document.querySelectorAll(`button[data-group="${group}"]`)[0].click();
        }
      });
    };

    const refreshInterval = 4000;
    const quotesBlocks = document.querySelectorAll('.quotes-list-viewport');

    if (quotesBlocks.length) {
      quotesBlocks.forEach(handleQuotesBlock);
      setInterval(refreshQuotesBlock, refreshInterval);
    }
  })();
</script>

      <script type="text/javascript" charset="utf-8">
  (() => {
    const doInit = (accordionRoot, idx) => {
      const baseID = `page-accordion-${idx}`;
      accordionRoot.id = baseID;
      const headings = accordionRoot.querySelectorAll('.accordion-header');
      if (!headings.length) {
        return;
      }

      const collapseElements = accordionRoot.querySelectorAll('.accordion-collapse');

      headings.forEach((headingElement, idx) => {
        const headingID = `${baseID}-heading-${idx}`;
        const targetID = `${baseID}-target-${idx}`;
        headingElement.id = headingID;
        const headingButton = headingElement.querySelector('.accordion-button');
        if (!headingButton) {
          return;
        }
        headingButton.type = 'button';
        headingButton.dataset['bsToggle'] = 'collapse';
        headingButton.dataset['bsTarget'] = `#${targetID}`;
        headingButton.setAttribute('aria-expanded', true);
        headingButton.setAttribute('aria-controls', targetID);
        headingButton.setAttribute('role', 'button');

        if (!collapseElements[idx].classList.contains('show')) {
          headingButton.classList.add('collapsed');
        }

        collapseElements[idx].id = targetID;
        collapseElements[idx].setAttribute('aria-labelledby', headingID);
      });

      new bootstrap.Collapse(accordionRoot);
    };

    const initAccordions = () => {
      const accordions = document.querySelectorAll('section.page__section div.accordion');
      if (!accordions.length) {
        return;
      }

      let accordionIndex = 0;
      accordions.forEach((accordion) => {
        doInit(accordion, accordionIndex);
        accordionIndex += 1;
      });
    };
    
    document.addEventListener('DOMContentLoaded', initAccordions)
  })();
</script>

      <script src="https://developer.nvidia.com/assets/grapesjs-tabs-f0b094476ecf56695b765f533e437303138b1e0824d993c50ff672e16dcccd8f.js"></script>
      <script src="https://developer.nvidia.com/assets/grapesjs-code-container-d9f9220defaa8a53ba3e8b7a6b1d155f6a1e5e5b37b9fa2282a7ace75bc60ae2.js"></script>
      <script src="https://developer.nvidia.com/assets/legacy-chart/d3.v4.min-41cfecdf7c41476e805de7afacf4aacdd1a4be6947fbecf95217e947ebc2faf5.js"></script>
      <script src="https://developer.nvidia.com/assets/legacy-chart/visualize-d-6a082dca95b2facd89f05ef138213cffea2419e9195042f6ceed4aa706943cb1.js"></script>
      <script src="https://developer.nvidia.com/assets/momentjs/moment-b955adb4137f92dd932ff2c3179ce60cb5e1daed5fcc4423f95cf17df02b4d68.js"></script>
      <script src="https://developer.nvidia.com/assets/momentjs/moment-timezone-with-data-10-year-range-dd05517070a46fa0052f9e706803d57a4fc38c1a223137ab480369e6308ba8d4.js"></script>
      <script src="https://developer.nvidia.com/assets/calendar-256ba38a1da92b24c057388ff6623eddd4cf1498f51d1a389cc4dfac501ab87c.js"></script>
      <script src="https://developer.nvidia.com/assets/hoverable-tooltip-06658c1c074450575999f586a2be4df71fb000fa8fbc3eec0955c1484012c593.js"></script>

    <script src="https://developer.nvidia.com/assets/nv-developer-menu-c88ee4614adbf7a100e9a74ffafb4d2c78601cff8c61752d42de9d8ba1b5d769.js"></script>
  <div id='footer' class='mt-auto'></div>
  <script src="https://www.nvidia.com/content/dam/en-zz/Solutions/librarian/bundle-search-prod-pub-v3.1.js"></script>
  <div id="librarian-overlay"></div>
  <script defer>
    setTimeout(() => {
      LIBRARIAN.Home.mount({
        elementId: 'librarian-overlay',
        searchPage: false,
        placeholder: '',
        site: 'https://developer.nvidia.com',
        retainFilters: true,
        overlay: true,
        pillsOpen: true,
        suggestedSearchPills: []
      })
    }, 1000)
  </script>
<script>
  let menuLocale = 'en';

  if (menuLocale == 'en') {
    menuLocale = 'en-US';
  }

  function mountHeader(data = false) {
    const headerTarget = document.getElementById('header');
    if (!headerTarget) {
      return;
    }
    let options = {
      baseURL: window.location.origin,
      signedIn: false,
      locale: menuLocale
    };

    if (data) {
      options.secondaryMenu = data;
    }


    options.showMembershipCardLink = true;

    new NVDeveloperHeader({
      target: headerTarget,
      props: options
    });
  }

  function mountFooter(data = false) {
    const footerTarget = document.getElementById('footer');
    if (!footerTarget) {
      return;
    }
    let options = {
      menu: data,
      locale: menuLocale
    };

    new NVDeveloperFooter({
      target: document.getElementById('footer'),
      props: options
    });
  }

  let url = 'd29g4g2dyqv443.cloudfront.net';
  let headerMenuURL = "https://d29g4g2dyqv443.cloudfront.net/menu/en-US/header-secondary.json";

  if (headerMenuURL) {
    fetch(headerMenuURL)
      .then(response => response.json())
      .then(data => {
        mountHeader(data);
      })
      .catch((error) => {
        mountHeader();
        window.nv.tracing.addError('menu', error);
      });
  } else {
    mountHeader();
  }

  fetch(`https://${url}/menu/${menuLocale}/footer.json`)
    .then(response => response.json())
    .then(data => {
      mountFooter(data);
    })
    .catch((error) => {
      mountFooter();
      window.nv.tracing.addError('menu', error);
    });
</script>

      <script src="https://www.datadoghq-browser-agent.com/us1/v5/datadog-rum.js"></script>

    
    
      <script>
  let silentAuthHost = 'www.nvidia.com';
  let crossOriginPageUrl = `https://${silentAuthHost}/auth/hints/`;

  function readHint() {
    return new Promise((resolve) => {
      const { origin: targetOrigin } = new URL(crossOriginPageUrl);

      const iframe = document.createElement('iframe');
      iframe.hidden = true;
      iframe.src = crossOriginPageUrl;

      function responseHandler(event) {
        if (event.origin === targetOrigin) {
          iframe.parentNode.removeChild(iframe);
          return resolve(event.data);
        }
      }

      window.addEventListener('message', responseHandler, { once: true });

      iframe.onload = () => {
        iframe.contentWindow.postMessage({ type: 'read' }, targetOrigin);
      }

      document.body.appendChild(iframe);
    });
  }

  function writeHint(login_hint, idp_id, timestamp, sub) {
    const { origin: targetOrigin } = new URL(crossOriginPageUrl);

    const iframe = document.createElement('iframe');
    iframe.hidden = true;
    iframe.src = crossOriginPageUrl;

    iframe.onload = () => {
      const message = { type: 'write', login_hint, idp_id, timestamp, sub };
      iframe.contentWindow.postMessage(message, targetOrigin);
    }

    document.body.appendChild(iframe);
  }

  function deleteHint() {
    const { origin: targetOrigin } = new URL(crossOriginPageUrl);

    const iframe = document.createElement('iframe');
    iframe.hidden = true;
    iframe.src = crossOriginPageUrl;

    iframe.onload = () => {
      iframe.contentWindow.postMessage({ type: 'delete' }, targetOrigin);
    }

    document.body.appendChild(iframe);
  }


</script>

    <script>_satellite.pageBottom();</script>
    <script src="https://api-prod.nvidia.com/search/nvidia-search-library.js"></script>

    <script src="https://developer.nvidia.com/assets/nv-gallery-widget-d1571cad15d961fd3ba013f57119be04a2adf32aa31b16be44a0f10f4671c37b.js"></script>
    <script src="https://developer.nvidia.com/packs/js/runtime-69f8104525d1541ca92d.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/4692-a478c93816dbe985b4fa.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/2681-d317b4e683473b6fe74d.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/9878-ace6038de3dd10e07189.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/8465-34d1ba762d9cdeef15fb.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/4622-0f5e83f2067789287ec7.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/8726-22e112d2c4254c5e6617.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/486-4ee95e78721864401fdd.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/8311-bc04e55c282e625862a2.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/1252-8ecaa37a655b20945baa.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/8823-1bca151f7e292c3ba6e8.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/2104-99d3913191dc4c0f1b36.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/6900-06759c19ec8c9a99401a.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/application-054e9f741655f1e87030.js" defer="defer"></script>
<script src="https://developer.nvidia.com/packs/js/ls_track-2e73cf990bfafea2d9d1.js" defer="defer"></script>
  </body>
</html>
