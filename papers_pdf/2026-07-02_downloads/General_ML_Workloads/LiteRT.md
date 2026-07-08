








<!doctype html>
<html 
      lang="en"
      dir="ltr">
  <head>
    <meta name="google-signin-client-id" content="721724668570-nbkv1cfusk7kk4eni4pjvepaus73b13t.apps.googleusercontent.com"><meta name="google-signin-scope"
          content="profile email https://www.googleapis.com/auth/developerprofiles https://www.googleapis.com/auth/developerprofiles.award https://www.googleapis.com/auth/devprofiles.full_control.firstparty"><meta property="og:site_name" content="Google for Developers">
    <meta property="og:type" content="website"><meta name="theme-color" content="#1a73e8"><meta charset="utf-8">
    <meta content="IE=Edge" http-equiv="X-UA-Compatible">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    

    <link rel="manifest" href="/_pwa/developers/manifest.json"
          crossorigin="use-credentials">
    <link rel="preconnect" href="//www.gstatic.com" crossorigin>
    <link rel="preconnect" href="//fonts.googleapis.com" crossorigin>
    <link rel="preconnect" href="//www.google-analytics.com" crossorigin><link rel="stylesheet" href="//fonts.googleapis.com/css?family=Google+Sans:400,500|Roboto:400,400italic,500,500italic,700,700italic|Roboto+Mono:400,500,700&display=swap">
      <link rel="stylesheet"
            href="//fonts.googleapis.com/css2?family=Material+Icons&family=Material+Symbols+Outlined&display=block"><link rel="stylesheet" href="https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers/css/app.css">
      <link rel="shortcut icon" href="https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers/images/favicon-new.png">
    <link rel="apple-touch-icon" href="https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers/images/touchicon-180-new.png"><link rel="canonical" href="https://developers.google.com/edge/litert"><link rel="search" type="application/opensearchdescription+xml"
            title="Google for Developers" href="https://developers.google.com/s/opensearch.xml">
      <link rel="alternate" hreflang="en"
          href="https://developers.google.com/edge/litert" /><link rel="alternate" hreflang="x-default" href="https://developers.google.com/edge/litert" /><title>LiteRT: High-Performance On-Device Machine Learning Framework &nbsp;|&nbsp; Google AI Edge &nbsp;|&nbsp; Google for Developers</title>

<meta property="og:title" content="LiteRT: High-Performance On-Device Machine Learning Framework &nbsp;|&nbsp; Google AI Edge &nbsp;|&nbsp; Google for Developers"><meta name="description" content="Seamlessly deploy GenAI and ML models on billions of devices with Google&#39;s high-performance framework.">
  <meta property="og:description" content="Seamlessly deploy GenAI and ML models on billions of devices with Google&#39;s high-performance framework."><meta name="keywords"
          
          content="litert, edgeai, ondeviceai, tflite, genai, tensorflowlite, aiagent, ml-runtime, edge-computing, mobile-ml"><meta name="category"
          
          content="AI/ML Development"><meta name="tool"
          
          content="LiteRT CLI, LiteRT SDK"><meta 
          property="og:title"
          content="LiteRT: Google&#39;s next-gen runtime for on-device AI"><meta 
          property="og:description"
          content="Unified, high-performance framework for deploying ML and GenAI on billions of edge devices."><meta 
          property="og:image"
          content="/edge/litert/images/landing/LiteRT_Blog1.jpg"><meta 
          property="og:type"
          content="website"><meta name="twitter:card"
          
          content="summary_large_image"><meta property="og:url" content="https://developers.google.com/edge/litert"><meta property="og:image" content="https://developers.google.com/static/edge/images/share.png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="675"><meta property="og:locale" content="en"><meta name="twitter:card" content="summary_large_image"><script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    "itemListElement": [{
      "@type": "ListItem",
      "position": 1,
      "name": "Google AI Edge",
      "item": "https://developers.google.com/edge"
    },{
      "@type": "ListItem",
      "position": 2,
      "name": "LiteRT: High-Performance On-Device Machine Learning Framework",
      "item": "https://developers.google.com/edge/litert"
    }]
  }
  </script>
    </head>
  <body class="color-scheme--light"
        template="landing"
        theme="google-blue"
        type="landing"
        
        
        
        layout="docs"
        
        
        
        
        
          
            concierge='hide'
          
        
        
        pending>
  
    <devsite-progress type="indeterminate" id="app-progress"></devsite-progress>
  
  
    <a href="#main-content" class="skip-link button">
      
      Skip to main content
    </a>
    <section class="devsite-wrapper">
      <devsite-cookie-notification-bar></devsite-cookie-notification-bar>
        <devsite-header role="banner">
  
    





















<div class="devsite-header--inner" data-nosnippet>
  <div class="devsite-top-logo-row-wrapper-wrapper">
    <div class="devsite-top-logo-row-wrapper">
      <div class="devsite-top-logo-row">
        <button type="button" id="devsite-hamburger-menu"
          class="devsite-header-icon-button button-flat material-icons gc-analytics-event"
          data-category="Site-Wide Custom Events"
          data-label="Navigation menu button"
          visually-hidden
          aria-label="Open menu">
        </button>
        
<div class="devsite-product-name-wrapper">

  <a href="/" class="devsite-site-logo-link gc-analytics-event"
   data-category="Site-Wide Custom Events" data-label="Site logo" track-type="globalNav"
   track-name="googleForDevelopers" track-metadata-position="nav"
   track-metadata-eventDetail="nav">
  
  <picture>
    
    <img src="https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers/images/lockup-new.svg" class="devsite-site-logo" alt="Google for Developers">
  </picture>
  
</a>



</div>
        <div class="devsite-top-logo-row-middle">
          <div class="devsite-header-upper-tabs">
            
           </div>
          
<devsite-search
    enable-signin
    enable-search
    enable-suggestions
      enable-query-completion
    
    enable-search-summaries
    project-name="Google AI Edge"
    tenant-name="Google for Developers"
    project-scope="/edge"
    url-scoped="https://developers.google.com/s/results/edge"
    
    
    
    >
  <form class="devsite-search-form" action="https://developers.google.com/s/results" method="GET">
    <div class="devsite-search-container">
      <button type="button"
              search-open
              class="devsite-search-button devsite-header-icon-button button-flat material-icons"
              
              aria-label="Open search"></button>
      <div class="devsite-searchbox">
        <input
          aria-activedescendant=""
          aria-autocomplete="list"
          
          aria-label="Search"
          aria-expanded="false"
          aria-haspopup="listbox"
          autocomplete="off"
          class="devsite-search-field devsite-search-query"
          name="q"
          
          placeholder="Search"
          role="combobox"
          type="text"
          value=""
          >
          <div class="devsite-search-image material-icons" aria-hidden="true">
            
              <svg class="devsite-search-ai-image" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <g clip-path="url(#clip0_6641_386)">
                    <path d="M19.6 21L13.3 14.7C12.8 15.1 12.225 15.4167 11.575 15.65C10.925 15.8833 10.2333 16 9.5 16C7.68333 16 6.14167 15.375 4.875 14.125C3.625 12.8583 3 11.3167 3 9.5C3 7.68333 3.625 6.15 4.875 4.9C6.14167 3.63333 7.68333 3 9.5 3C10.0167 3 10.5167 3.05833 11 3.175C11.4833 3.275 11.9417 3.43333 12.375 3.65L10.825 5.2C10.6083 5.13333 10.3917 5.08333 10.175 5.05C9.95833 5.01667 9.73333 5 9.5 5C8.25 5 7.18333 5.44167 6.3 6.325C5.43333 7.19167 5 8.25 5 9.5C5 10.75 5.43333 11.8167 6.3 12.7C7.18333 13.5667 8.25 14 9.5 14C10.6667 14 11.6667 13.625 12.5 12.875C13.35 12.1083 13.8417 11.15 13.975 10H15.975C15.925 10.6333 15.7833 11.2333 15.55 11.8C15.3333 12.3667 15.05 12.8667 14.7 13.3L21 19.6L19.6 21ZM17.5 12C17.5 10.4667 16.9667 9.16667 15.9 8.1C14.8333 7.03333 13.5333 6.5 12 6.5C13.5333 6.5 14.8333 5.96667 15.9 4.9C16.9667 3.83333 17.5 2.53333 17.5 0.999999C17.5 2.53333 18.0333 3.83333 19.1 4.9C20.1667 5.96667 21.4667 6.5 23 6.5C21.4667 6.5 20.1667 7.03333 19.1 8.1C18.0333 9.16667 17.5 10.4667 17.5 12Z" fill="#5F6368"/>
                  </g>
                <defs>
                <clipPath id="clip0_6641_386">
                <rect width="24" height="24" fill="white"/>
                </clipPath>
                </defs>
              </svg>
            
          </div>
          <div class="devsite-search-shortcut-icon-container" aria-hidden="true">
            <kbd class="devsite-search-shortcut-icon">/</kbd>
          </div>
      </div>
    </div>
  </form>
  <button type="button"
          search-close
          class="devsite-search-button devsite-header-icon-button button-flat material-icons"
          
          aria-label="Close search"></button>
</devsite-search>

        </div>

        

          

          

          

          

          
<devsite-language-selector>
  <ul role="presentation">
    
    
    <li role="presentation">
      <a role="menuitem" lang="en"
        >English</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="de"
        >Deutsch</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="es"
        >Español</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="fr"
        >Français</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="id"
        >Indonesia</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="pt_br"
        >Português – Brasil</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="ru"
        >Русский</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="zh_cn"
        >中文 – 简体</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="ja"
        >日本語</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="ko"
        >한국어</a>
    </li>
    
  </ul>
</devsite-language-selector>


          

        

        
          <devsite-user 
                        
                        
                          enable-profiles
                        
                        
                          fp-auth
                        
                        id="devsite-user">
            
              
              <span class="button devsite-top-button" aria-hidden="true" visually-hidden>Sign in</span>
            
          </devsite-user>
        
        
        
      </div>
    </div>
  </div>



  <div class="devsite-collapsible-section
    ">
    <div class="devsite-header-background">
      
        
          <div class="devsite-product-id-row"
           >
            <div class="devsite-product-description-row">
              
                
                <div class="devsite-product-id">
                  
                  
                  
                    <ul class="devsite-breadcrumb-list"
  >
  
  <li class="devsite-breadcrumb-item
             ">
    
    
    
      
        
  <a href="https://developers.google.com/edge"
      
        class="devsite-breadcrumb-link gc-analytics-event"
      
        data-category="Site-Wide Custom Events"
      
        data-label="Lower Header"
      
        data-value="1"
      
        track-type="globalNav"
      
        track-name="breadcrumb"
      
        track-metadata-position="1"
      
        track-metadata-eventdetail="Google AI Edge"
      
    >
    
          Google AI Edge
        
  </a>
  
      
    
  </li>
  
</ul>
                </div>
                
              
              
            </div>
            
          </div>
          
        
      
      
        <div class="devsite-doc-set-nav-row">
          
          
            
            
  <devsite-tabs class="lower-tabs">

    <nav class="devsite-tabs-wrapper" aria-label="Lower tabs">
      
        
          <tab class="devsite-dropdown
    
    devsite-active
    
    ">
  
    <a href="https://developers.google.com/edge/litert"
    class="devsite-tabs-content gc-analytics-event "
      track-metadata-eventdetail="https://developers.google.com/edge/litert"
    
       track-type="nav"
       track-metadata-position="nav - litert"
       track-metadata-module="primary nav"
       aria-label="LiteRT, selected" 
       
         
           data-category="Site-Wide Custom Events"
         
           data-label="Tab: LiteRT"
         
           track-name="litert"
         
       >
    LiteRT
  
    </a>
    
      <button
         aria-haspopup="menu"
         aria-expanded="false"
         aria-label="Dropdown menu for LiteRT"
         track-type="nav"
         track-metadata-eventdetail="https://developers.google.com/edge/litert"
         track-metadata-position="nav - litert"
         track-metadata-module="primary nav"
         
          
            data-category="Site-Wide Custom Events"
          
            data-label="Tab: LiteRT"
          
            track-name="litert"
          
        
         class="devsite-tabs-dropdown-toggle devsite-icon devsite-icon-arrow-drop-down"></button>
    
  
  <div class="devsite-tabs-dropdown" role="menu" aria-label="submenu" hidden>
    <div class="devsite-tabs-dropdown-content">
      
      
        <div class="devsite-tabs-dropdown-column
                    ">
          
            <ul class="devsite-tabs-dropdown-section
                       ">
              
              
              
                <li class="devsite-nav-item">
                  <a href="https://developers.google.com/edge/litert/android"
                    
                     track-type="nav"
                     track-metadata-eventdetail="https://developers.google.com/edge/litert/android"
                     track-metadata-position="nav - litert"
                     track-metadata-module="tertiary nav"
                     
                     tooltip
                  >
                    
                    <div class="devsite-nav-item-title">
                      Android
                    </div>
                    
                  </a>
                </li>
              
                <li class="devsite-nav-item">
                  <a href="https://developers.google.com/edge/litert/next/python"
                    
                     track-type="nav"
                     track-metadata-eventdetail="https://developers.google.com/edge/litert/next/python"
                     track-metadata-position="nav - litert"
                     track-metadata-module="tertiary nav"
                     
                     tooltip
                  >
                    
                    <div class="devsite-nav-item-title">
                      Desktop
                    </div>
                    
                  </a>
                </li>
              
                <li class="devsite-nav-item">
                  <a href="https://developers.google.com/edge/litert/web"
                    
                     track-type="nav"
                     track-metadata-eventdetail="https://developers.google.com/edge/litert/web"
                     track-metadata-position="nav - litert"
                     track-metadata-module="tertiary nav"
                     
                     tooltip
                  >
                    
                    <div class="devsite-nav-item-title">
                      Web
                    </div>
                    
                  </a>
                </li>
              
            </ul>
          
        </div>
      
    </div>
  </div>
</tab>
        
      
        
          <tab  >
            
    <a href="https://developers.google.com/edge/litert-lm"
    class="devsite-tabs-content gc-analytics-event "
      track-metadata-eventdetail="https://developers.google.com/edge/litert-lm"
    
       track-type="nav"
       track-metadata-position="nav - litert-lm"
       track-metadata-module="primary nav"
       
       
         
           data-category="Site-Wide Custom Events"
         
           data-label="Tab: LiteRT-LM"
         
           track-name="litert-lm"
         
       >
    LiteRT-LM
  
    </a>
    
  
          </tab>
        
      
        
          <tab class="devsite-dropdown
    
    
    
    ">
  
    <a href="https://developers.google.com/edge/mediapipe/solutions/guide"
    class="devsite-tabs-content gc-analytics-event "
      track-metadata-eventdetail="https://developers.google.com/edge/mediapipe/solutions/guide"
    
       track-type="nav"
       track-metadata-position="nav - mediapipe"
       track-metadata-module="primary nav"
       
       
         
           data-category="Site-Wide Custom Events"
         
           data-label="Tab: MediaPipe"
         
           track-name="mediapipe"
         
       >
    MediaPipe
  
    </a>
    
      <button
         aria-haspopup="menu"
         aria-expanded="false"
         aria-label="Dropdown menu for MediaPipe"
         track-type="nav"
         track-metadata-eventdetail="https://developers.google.com/edge/mediapipe/solutions/guide"
         track-metadata-position="nav - mediapipe"
         track-metadata-module="primary nav"
         
          
            data-category="Site-Wide Custom Events"
          
            data-label="Tab: MediaPipe"
          
            track-name="mediapipe"
          
        
         class="devsite-tabs-dropdown-toggle devsite-icon devsite-icon-arrow-drop-down"></button>
    
  
  <div class="devsite-tabs-dropdown" role="menu" aria-label="submenu" hidden>
    <div class="devsite-tabs-dropdown-content">
      
      
        <div class="devsite-tabs-dropdown-column
                    ">
          
            <ul class="devsite-tabs-dropdown-section
                       ">
              
              
              
                <li class="devsite-nav-item">
                  <a href="https://developers.google.com/edge/mediapipe/solutions/guide"
                    
                     track-type="nav"
                     track-metadata-eventdetail="https://developers.google.com/edge/mediapipe/solutions/guide"
                     track-metadata-position="nav - mediapipe"
                     track-metadata-module="tertiary nav"
                     
                     tooltip
                  >
                    
                    <div class="devsite-nav-item-title">
                      MediaPipe Solutions
                    </div>
                    
                  </a>
                </li>
              
                <li class="devsite-nav-item">
                  <a href="https://developers.google.com/edge/mediapipe/framework"
                    
                     track-type="nav"
                     track-metadata-eventdetail="https://developers.google.com/edge/mediapipe/framework"
                     track-metadata-position="nav - mediapipe"
                     track-metadata-module="tertiary nav"
                     
                     tooltip
                  >
                    
                    <div class="devsite-nav-item-title">
                      MediaPipe Framework
                    </div>
                    
                  </a>
                </li>
              
            </ul>
          
        </div>
      
    </div>
  </div>
</tab>
        
      
        
          <tab  >
            
    <a href="https://developers.google.com/edge/model-explorer"
    class="devsite-tabs-content gc-analytics-event "
      track-metadata-eventdetail="https://developers.google.com/edge/model-explorer"
    
       track-type="nav"
       track-metadata-position="nav - model explorer"
       track-metadata-module="primary nav"
       
       
         
           data-category="Site-Wide Custom Events"
         
           data-label="Tab: Model Explorer"
         
           track-name="model explorer"
         
       >
    Model Explorer
  
    </a>
    
  
          </tab>
        
      
        
          <tab  >
            
    <a href="https://developers.google.com/edge/ai-edge-portal"
    class="devsite-tabs-content gc-analytics-event "
      track-metadata-eventdetail="https://developers.google.com/edge/ai-edge-portal"
    
       track-type="nav"
       track-metadata-position="nav - ai edge portal"
       track-metadata-module="primary nav"
       
       
         
           data-category="Site-Wide Custom Events"
         
           data-label="Tab: AI Edge Portal"
         
           track-name="ai edge portal"
         
       >
    AI Edge Portal
  
    </a>
    
  
          </tab>
        
      
        
          <tab  >
            
    <a href="https://developers.google.com/edge/tensor-sdk"
    class="devsite-tabs-content gc-analytics-event "
      track-metadata-eventdetail="https://developers.google.com/edge/tensor-sdk"
    
       track-type="nav"
       track-metadata-position="nav - google tensor sdk"
       track-metadata-module="primary nav"
       
       
         
           data-category="Site-Wide Custom Events"
         
           data-label="Tab: Google Tensor SDK"
         
           track-name="google tensor sdk"
         
       >
    Google Tensor SDK
  
    </a>
    
  
          </tab>
        
      
        
          <tab class="devsite-dropdown
    
    
    
    ">
  
    <button
      class="devsite-tabs-content devsite-tabs-dropdown-only gc-analytics-event  devsite-icon devsite-icon-arrow-drop-down"
  
       track-type="nav"
       track-metadata-position="nav - apps"
       track-metadata-module="primary nav"
       
       
         
           data-category="Site-Wide Custom Events"
         
           data-label="Tab: Apps"
         
           track-name="apps"
         
       >
    Apps
  
  </button>
  
  <div class="devsite-tabs-dropdown" role="menu" aria-label="submenu" hidden>
    <div class="devsite-tabs-dropdown-content">
      
      
        <div class="devsite-tabs-dropdown-column
                    ">
          
            <ul class="devsite-tabs-dropdown-section
                       ">
              
              
              
                <li class="devsite-nav-item">
                  <a href="https://developers.google.com/edge/gallery"
                    
                     track-type="nav"
                     track-metadata-eventdetail="https://developers.google.com/edge/gallery"
                     track-metadata-position="nav - apps"
                     track-metadata-module="tertiary nav"
                     
                     tooltip
                  >
                    
                    <div class="devsite-nav-item-title">
                      AI Edge Gallery
                    </div>
                    
                  </a>
                </li>
              
                <li class="devsite-nav-item">
                  <a href="https://developers.google.com/edge/eloquent"
                    
                     track-type="nav"
                     track-metadata-eventdetail="https://developers.google.com/edge/eloquent"
                     track-metadata-position="nav - apps"
                     track-metadata-module="tertiary nav"
                     
                     tooltip
                  >
                    
                    <div class="devsite-nav-item-title">
                      AI Edge Eloquent
                    </div>
                    
                  </a>
                </li>
              
                <li class="devsite-nav-item">
                  <a href="https://google-ai-edge.github.io/mediapipe-samples-web"
                    
                     track-type="nav"
                     track-metadata-eventdetail="https://google-ai-edge.github.io/mediapipe-samples-web"
                     track-metadata-position="nav - apps"
                     track-metadata-module="tertiary nav"
                     
                     tooltip
                  >
                    
                    <div class="devsite-nav-item-title">
                      MediaPipe Demos
                    </div>
                    
                  </a>
                </li>
              
            </ul>
          
        </div>
      
    </div>
  </div>
</tab>
        
      
        
          <tab  >
            
    <a href="https://developers.google.com/edge/api"
    class="devsite-tabs-content gc-analytics-event "
      track-metadata-eventdetail="https://developers.google.com/edge/api"
    
       track-type="nav"
       track-metadata-position="nav - api reference"
       track-metadata-module="primary nav"
       
       
         
           data-category="Site-Wide Custom Events"
         
           data-label="Tab: API Reference"
         
           track-name="api reference"
         
       >
    API Reference
  
    </a>
    
  
          </tab>
        
      
    </nav>

  </devsite-tabs>

          
          
        </div>
      
    </div>
  </div>

</div>



  

  
</devsite-header>
        <devsite-book-nav scrollbars >
          
            





















<div class="devsite-book-nav-filter"
     hidden>
  <span class="filter-list-icon material-icons" aria-hidden="true"></span>
  <input type="text"
         placeholder="Filter"
         
         aria-label="Type to filter"
         role="searchbox">
  
  <span class="filter-clear-button hidden"
        data-title="Clear filter"
        aria-label="Clear filter"
        role="button"
        tabindex="0"></span>
</div>

<nav class="devsite-book-nav devsite-nav nocontent" data-nosnippet
     aria-label="Side menu">
  <div class="devsite-mobile-header">
    <button type="button"
            id="devsite-close-nav"
            class="devsite-header-icon-button button-flat material-icons gc-analytics-event"
            data-category="Site-Wide Custom Events"
            data-label="Close navigation"
            aria-label="Close navigation">
    </button>
    <div class="devsite-product-name-wrapper">

  <a href="/" class="devsite-site-logo-link gc-analytics-event"
   data-category="Site-Wide Custom Events" data-label="Site logo" track-type="globalNav"
   track-name="googleForDevelopers" track-metadata-position="nav"
   track-metadata-eventDetail="nav">
  
  <picture>
    
    <img src="https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers/images/lockup-new.svg" class="devsite-site-logo" alt="Google for Developers">
  </picture>
  
</a>


</div>
  </div>

  <div class="devsite-book-nav-wrapper">
    <div class="devsite-mobile-nav-top">
      
        <ul class="devsite-nav-list">
          
            <li class="devsite-nav-item">
              
  
  <a href="/edge"
    
       class="devsite-nav-title gc-analytics-event
              
              devsite-nav-active"
    

    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: AI Edge"
      
        track-name="ai edge"
      
    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: AI Edge"
     track-type="globalNav"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      AI Edge
   </span>
    
  
  </a>
  

  
              
                <ul class="devsite-nav-responsive-tabs">
                  
                    
                    
                    
                    <li class="devsite-nav-item">
                      
  
  <a href="/edge/litert"
    
       class="devsite-nav-title gc-analytics-event
              devsite-nav-has-children
              devsite-nav-active"
    

    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: LiteRT"
      
        track-name="litert"
      
    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: LiteRT"
     track-type="globalNav"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip menu="_book">
      LiteRT
   </span>
    
    <span class="devsite-nav-icon material-icons" data-icon="forward"
          menu="_book">
    </span>
    
  
  </a>
  

  
    <ul class="devsite-nav-responsive-tabs devsite-nav-has-menu
                devsite-lower-tab-item">
      
<li class="devsite-nav-item">

  
  <span
    
       class="devsite-nav-title"
       tooltip
    
    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: LiteRT"
      
        track-name="litert"
      
    >
  
    <span class="devsite-nav-text" tooltip menu="LiteRT">
      More
   </span>
    
    <span class="devsite-nav-icon material-icons" data-icon="forward"
          menu="LiteRT">
    </span>
    
  
  </span>
  

</li>

    </ul>
  
                    </li>
                  
                    
                    
                    
                    <li class="devsite-nav-item">
                      
  
  <a href="/edge/litert-lm"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: LiteRT-LM"
      
        track-name="litert-lm"
      
    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: LiteRT-LM"
     track-type="globalNav"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      LiteRT-LM
   </span>
    
  
  </a>
  

  
                    </li>
                  
                    
                    
                    
                    <li class="devsite-nav-item">
                      
  
  <a href="/edge/mediapipe/solutions/guide"
    
       class="devsite-nav-title gc-analytics-event
              devsite-nav-has-children
              "
    

    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: MediaPipe"
      
        track-name="mediapipe"
      
    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: MediaPipe"
     track-type="globalNav"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      MediaPipe
   </span>
    
    <span class="devsite-nav-icon material-icons" data-icon="forward"
          >
    </span>
    
  
  </a>
  

  
    <ul class="devsite-nav-responsive-tabs devsite-nav-has-menu
                devsite-lower-tab-item">
      
<li class="devsite-nav-item">

  
  <span
    
       class="devsite-nav-title"
       tooltip
    
    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: MediaPipe"
      
        track-name="mediapipe"
      
    >
  
    <span class="devsite-nav-text" tooltip menu="MediaPipe">
      More
   </span>
    
    <span class="devsite-nav-icon material-icons" data-icon="forward"
          menu="MediaPipe">
    </span>
    
  
  </span>
  

</li>

    </ul>
  
                    </li>
                  
                    
                    
                    
                    <li class="devsite-nav-item">
                      
  
  <a href="/edge/model-explorer"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: Model Explorer"
      
        track-name="model explorer"
      
    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: Model Explorer"
     track-type="globalNav"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      Model Explorer
   </span>
    
  
  </a>
  

  
                    </li>
                  
                    
                    
                    
                    <li class="devsite-nav-item">
                      
  
  <a href="/edge/ai-edge-portal"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: AI Edge Portal"
      
        track-name="ai edge portal"
      
    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: AI Edge Portal"
     track-type="globalNav"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      AI Edge Portal
   </span>
    
  
  </a>
  

  
                    </li>
                  
                    
                    
                    
                    <li class="devsite-nav-item">
                      
  
  <a href="/edge/tensor-sdk"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: Google Tensor SDK"
      
        track-name="google tensor sdk"
      
    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: Google Tensor SDK"
     track-type="globalNav"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      Google Tensor SDK
   </span>
    
  
  </a>
  

  
                    </li>
                  
                    
                    
                    
                    <li class="devsite-nav-item">
                      
  
  <span
    
       class="devsite-nav-title"
       tooltip
    
    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: Apps"
      
        track-name="apps"
      
    >
  
    <span class="devsite-nav-text" tooltip >
      Apps
   </span>
    
  
  </span>
  

  
    <ul class="devsite-nav-responsive-tabs devsite-nav-has-menu
                devsite-lower-tab-item">
      
<li class="devsite-nav-item">

  
  <span
    
       class="devsite-nav-title"
       tooltip
    
    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: Apps"
      
        track-name="apps"
      
    >
  
    <span class="devsite-nav-text" tooltip menu="Apps">
      More
   </span>
    
    <span class="devsite-nav-icon material-icons" data-icon="forward"
          menu="Apps">
    </span>
    
  
  </span>
  

</li>

    </ul>
  
                    </li>
                  
                    
                    
                    
                    <li class="devsite-nav-item">
                      
  
  <a href="/edge/api"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
      
        data-category="Site-Wide Custom Events"
      
        data-label="Tab: API Reference"
      
        track-name="api reference"
      
    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: API Reference"
     track-type="globalNav"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      API Reference
   </span>
    
  
  </a>
  

  
                    </li>
                  
                </ul>
              
            </li>
          
          
          
        </ul>
      
    </div>
    
      <div class="devsite-mobile-nav-bottom">
        
          
          <ul class="devsite-nav-list" menu="_book">
            <li class="devsite-nav-item"><a href="/edge/litert"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Home</span></a></li>

  <li class="devsite-nav-item"><a href="/edge/litert/overview"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li>

  <li class="devsite-nav-item"><a href="/edge/litert/migration"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Migrating from TensorFlow Lite</span></a></li>

  <li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>LiteRT CLI</span>
      </div></li>

  <li class="devsite-nav-item"><a href="/edge/litert/cli"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li>

  <li class="devsite-nav-item"><a href="/edge/litert/cli/installation"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Installation</span></a></li>

  <li class="devsite-nav-item"><a href="/edge/litert/cli/commands"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Common Commands</span></a></li>

  <li class="devsite-nav-item"><a href="/edge/litert/cli/troubleshooting"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Troubleshooting &amp; Resources</span></a></li>

  <li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>GenAI Deployments</span>
      </div></li>

  <li class="devsite-nav-item"><a href="/edge/litert/genai/overview"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li>

  <li class="devsite-nav-item"><a href="/edge/litert/conversion/pytorch/genai"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Convert PyTorch GenAI models</span></a></li>

  <li class="devsite-nav-item"><a href="/edge/litert/next/litert_lm_npu"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Run LLMs using LiteRT-LM</span></a></li>

  <li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Model Conversion &amp; Optimization</span>
      </div></li>

  <li class="devsite-nav-item"><a href="/edge/litert/conversion/overview"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Convert PyTorch models</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/conversion/pytorch/overview"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/pytorch/genai"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Convert PyTorch GenAI models</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Convert TensorFlow models</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/overview"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/pretrained_models"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Use pre-trained models</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/convert_tf"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Convert TensorFlow models</span></a></li><li class="devsite-nav-item
           devsite-nav-experimental"><a href="/edge/litert/conversion/tensorflow/signatures"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Add Signatures</span><span class="devsite-nav-icon material-icons"
        data-icon="experimental"
        data-title="Experimental!"
        aria-hidden="true"></span></a></li><li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Model compatibility</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/ops_compatibility"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/ops_select"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Select operators</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/op_select_allowlist"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Select operators Allowlist</span></a></li><li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Advanced</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/operation_fusion"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Fused operators</span></a></li><li class="devsite-nav-item
           devsite-nav-experimental"><a href="/edge/litert/conversion/tensorflow/ops_version"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Operator versions</span><span class="devsite-nav-icon material-icons"
        data-icon="experimental"
        data-title="Experimental!"
        aria-hidden="true"></span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/rnn"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>RNN models</span></a></li></ul></div></li></ul></div></li><li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Optimize models</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/quantization/model_optimization"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/quantization/post_training_quantization"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Post-training quantization</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/quantization/post_training_quant"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Post-training dynamic range quantization</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Post-training integer quantization</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/quantization/post_training_float16_quant"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Post-training float16 quantization</span></a></li><li class="devsite-nav-item
           devsite-nav-experimental"><a href="/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant_16x8"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Post-training integer quantization with int16 activations</span><span class="devsite-nav-icon material-icons"
        data-icon="experimental"
        data-title="Experimental!"
        aria-hidden="true"></span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/quantization/quantization_spec"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Quantization specification</span></a></li><li class="devsite-nav-item
           devsite-nav-nightly"><a href="/edge/litert/conversion/tensorflow/quantization/quantization_debugger"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Inspecting quantization errors</span><span class="devsite-nav-icon material-icons"
        data-icon="nightly"
        data-title="Nightly build only"
        aria-hidden="true"></span></a></li><li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Add model metadata</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/metadata"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/metadata_writer_tutorial"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Metadata Writer API</span></a></li></ul></div></li></ul></div></li><li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Design and build models</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/build/overview"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/build/best_practices"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Performance best practices</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/build/ondevice_training"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>On-device training</span></a></li></ul></div></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Convert JAX models</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/conversion/jax/overview"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Inference &amp; Hardware Acceleration</span>
      </div></li>

  <li class="devsite-nav-item"><a href="/edge/litert/inference"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li>

  <li class="devsite-nav-item"><a href="/edge/litert/next/gpu"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>GPU acceleration</span></a></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>NPU acceleration</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/next/npu"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/litert_lm_npu"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Run LLMs using LiteRT-LM</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/tensor-sdk"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Google Tensor</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/intel"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Intel</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/qualcomm"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Qualcomm</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/mediatek"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>MediaTek</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/samsung"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Samsung</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Creating a new accelerator</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/next/compiler_plugin"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Implementing a Compiler Plugin</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/dispatch"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Implementing the Dispatch API</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/ats"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Accelerator Test Suite (ATS)</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Benchmark &amp; Profiling</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/next/benchmark"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Benchmark CompiledModel API</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/models/measurement"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Benchmark Interpreter API</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Create custom operators</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/next/custom_op_dispatcher"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Custom ops of CompiledModel API</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/conversion/tensorflow/ops_custom"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Custom ops of Interpreter API</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Run on Android</span>
      </div></li>

  <li class="devsite-nav-item"><a href="/edge/litert/android"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Run with CompiledModel API (accelerated on GPU/NPU)</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/next/android_kotlin"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Kotlin API</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/cpp"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>C++ API</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/android_cpp_sdk"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Use prebuilt C++ library</span></a></li><li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Hardware acceleration</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/next/gpu"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>GPU acceleration</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/npu"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>NPU acceleration</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/tensor-sdk"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Google Tensor</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/mediatek"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>MediaTek NPU</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/qualcomm"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Qualcomm NPU</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/intel"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Intel NPU</span></a></li></ul></div></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Run with Interpreter API</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Google Play services runtime</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/android/play_services"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/android/java"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Java API</span></a></li><li class="devsite-nav-item
           devsite-nav-experimental"><a href="/edge/litert/android/native"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>C API</span><span class="devsite-nav-icon material-icons"
        data-icon="experimental"
        data-title="Experimental!"
        aria-hidden="true"></span></a></li></ul></div></li><li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Hardware acceleration</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item
           devsite-nav-experimental"><a href="/edge/litert/android/acceleration_service"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Acceleration service</span><span class="devsite-nav-icon material-icons"
        data-icon="experimental"
        data-title="Experimental!"
        aria-hidden="true"></span></a></li><li class="devsite-nav-item"><a href="/edge/litert/android/gpu"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>GPU with Interpreter API</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/android/gpu_native"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>GPU with C/C++ API</span></a></li><li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>NPU delegates</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/android/npu/overview"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/android/npu/qualcomm"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Qualcomm NPUs for Mobile AI Development</span></a></li></ul></div></li></ul></div></li><li class="devsite-nav-item"><a href="/edge/litert/android/development"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Development tools</span></a></li><li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Models with metadata</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/android/metadata/overview"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/android/metadata/codegen"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Generate model interfaces</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/android/metadata/lite_support"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Customize data input and output</span></a></li></ul></div></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Run on iOS / macOS</span>
      </div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Run with CompiledModel API (accelerated on GPU)</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/next/cpp"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>C++ API</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/cpp_sdk"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Use prebuilt C++ library</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Run with Interpreter API</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/ios/quickstart"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Swift API</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/ios/coreml"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Core ML delegate</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/ios/gpu"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>GPU delegate</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Run on Web with LiteRT.js</span>
      </div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Run with CompiledModel API (accelerated on GPU)</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/web"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/web/get_started"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>JavaScript API</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Run on Desktop (Linux, Windows)</span>
      </div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Run with CompiledModel API (accelerated on GPU)</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/next/cpp"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>C++ API</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/cpp_sdk"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Use prebuilt C++ library</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/python"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Python API</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Run on Embedded &amp; IoT</span>
      </div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Run with CompiledModel API</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/next/cpp"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>C++ API</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/next/cpp_sdk"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Use prebuilt C++ library</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Run with Interpreter API</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/microcontrollers/overview"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/microcontrollers/get_started"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Get started</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/microcontrollers/python"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Linux-based devices with Python</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/microcontrollers/library"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Understand the C++ library</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/microcontrollers/build_convert"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Build and convert models</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Build from Source</span>
      </div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Build Compiled Model API</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/build/cmake_litert"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Build with CMake</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Build Interpreter API</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/build/android"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Build for Android</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/build/ios"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Build for iOS</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/build/arm"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Build for Linux-based IoT</span></a></li><li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Build with CMake</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/build/cmake"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/build/cmake_arm"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Cross compilation for ARM</span></a></li></ul></div></li><li class="devsite-nav-item"><a href="/edge/litert/build/cmake_pip"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Build Python Wheel</span></a></li><li class="devsite-nav-item
           devsite-nav-experimental"><a href="/edge/litert/build/reduce_binary_size"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Reduce binary size</span><span class="devsite-nav-icon material-icons"
        data-icon="experimental"
        data-title="Experimental!"
        aria-hidden="true"></span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Libraries and tools</span>
      </div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Task Library</span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/overview"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/image_classifier"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>ImageClassifier</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/object_detector"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>ObjectDetector</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/image_segmenter"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>ImageSegmenter</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/image_embedder"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>ImageEmbedder</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/image_searcher"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>ImageSearcher</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/nl_classifier"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>NLClassifier</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/bert_nl_classifier"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>BertNLClassifier</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/bert_question_answerer"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>BertQuestionAnswerer</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/text_embedder"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>TextEmbedder</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/text_searcher"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>TextSearcher</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/audio_classifier"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>AudioClassifier</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/task_library/customized_task_api"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Customized API</span></a></li></ul></div></li>

  <li class="devsite-nav-item
           devsite-nav-expandable
           devsite-nav-experimental"><div class="devsite-expandable-nav">
      <a class="devsite-nav-toggle" aria-hidden="true"></a><div class="devsite-nav-title devsite-nav-title-no-path" tabindex="0" role="button">
        <span class="devsite-nav-text" tooltip>Model Maker</span><span class="devsite-nav-icon material-icons"
        data-icon="experimental"
        data-title="Experimental!"
        aria-hidden="true"></span>
      </div><ul class="devsite-nav-section"><li class="devsite-nav-item"><a href="/edge/litert/libraries/modify"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Overview</span></a></li><li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Images &amp; video</span>
      </div></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/modify/image_classification"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Image classification</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/modify/object_detection"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Object detection</span></a></li><li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Text</span>
      </div></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/modify/text_classification"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Text classification</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/modify/question_answer"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>BERT question &amp; answer</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/modify/text_searcher"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Text search</span></a></li><li class="devsite-nav-item
           devsite-nav-heading"><div class="devsite-nav-title devsite-nav-title-no-path">
        <span class="devsite-nav-text" tooltip>Audio</span>
      </div></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/modify/audio_classification"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Audio classification</span></a></li><li class="devsite-nav-item"><a href="/edge/litert/libraries/modify/speech_recognition"
        class="devsite-nav-title"
      ><span class="devsite-nav-text" tooltip>Speech recognition</span></a></li></ul></div></li>
          </ul>
        
        
          
    
  
        
        
          
    
      
      <ul class="devsite-nav-list" menu="LiteRT"
          aria-label="Side menu" hidden>
        
          
            
            
              
<li class="devsite-nav-item">

  
  <a href="/edge/litert/android"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: Android"
     track-type="navMenu"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      Android
   </span>
    
  
  </a>
  

</li>

            
              
<li class="devsite-nav-item">

  
  <a href="/edge/litert/next/python"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: Desktop"
     track-type="navMenu"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      Desktop
   </span>
    
  
  </a>
  

</li>

            
              
<li class="devsite-nav-item">

  
  <a href="/edge/litert/web"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: Web"
     track-type="navMenu"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      Web
   </span>
    
  
  </a>
  

</li>

            
          
        
      </ul>
    
  
    
  
    
      
      <ul class="devsite-nav-list" menu="MediaPipe"
          aria-label="Side menu" hidden>
        
          
            
            
              
<li class="devsite-nav-item">

  
  <a href="/edge/mediapipe/solutions/guide"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: MediaPipe Solutions"
     track-type="navMenu"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      MediaPipe Solutions
   </span>
    
  
  </a>
  

</li>

            
              
<li class="devsite-nav-item">

  
  <a href="/edge/mediapipe/framework"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: MediaPipe Framework"
     track-type="navMenu"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      MediaPipe Framework
   </span>
    
  
  </a>
  

</li>

            
          
        
      </ul>
    
  
    
  
    
  
    
  
    
      
      <ul class="devsite-nav-list" menu="Apps"
          aria-label="Side menu" hidden>
        
          
            
            
              
<li class="devsite-nav-item">

  
  <a href="/edge/gallery"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: AI Edge Gallery"
     track-type="navMenu"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      AI Edge Gallery
   </span>
    
  
  </a>
  

</li>

            
              
<li class="devsite-nav-item">

  
  <a href="/edge/eloquent"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: AI Edge Eloquent"
     track-type="navMenu"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      AI Edge Eloquent
   </span>
    
  
  </a>
  

</li>

            
              
<li class="devsite-nav-item">

  
  <a href="https://google-ai-edge.github.io/mediapipe-samples-web"
    
       class="devsite-nav-title gc-analytics-event
              
              "
    

    
     data-category="Site-Wide Custom Events"
     data-label="Responsive Tab: MediaPipe Demos"
     track-type="navMenu"
     track-metadata-eventDetail="globalMenu"
     track-metadata-position="nav">
  
    <span class="devsite-nav-text" tooltip >
      MediaPipe Demos
   </span>
    
  
  </a>
  

</li>

            
          
        
      </ul>
    
  
    
  
        
      </div>
    
  </div>
</nav>
          
        </devsite-book-nav>
      
      <section id="gc-wrapper">
        <main role="main" id="main-content" class="devsite-main-content"
            
              has-book-nav
              
            >
          <div class="devsite-sidebar">
            <div class="devsite-sidebar-content">
                
                <devsite-toc class="devsite-nav"
                            role="navigation"
                            aria-label="On this page"
                            depth="2"
                            scrollbars
                            data-nosnippet
                  disabled></devsite-toc>
                <devsite-recommendations-sidebar class="nocontent devsite-nav" data-nosnippet>
                </devsite-recommendations-sidebar>
            </div>
          </div>
          <devsite-content>
            
              










<article class="devsite-article"><style>
      /* Styles inlined from /edge/litert/css/landing.css */
/* LiteRT Modernized Landing Styles
 * ---------------------------------------------------------
 * This stylesheet implements premium on-device AI branding
 * for the LiteRT DevSite landing page.
 */

:root {
  /* Google Brand Palette */
  --lite-blue: #1a73e8;
  --lite-blue-hover: #1765cc;
  --lite-blue-focus: rgba(26, 115, 232, 0.24);
  --lite-accent: #4285f4;

  /* Neutrals & Surfaces */
  --lite-grey-bg: #f8f9fa;
  --lite-border: rgba(60, 64, 67, 0.12);
  --lite-text-primary: #202124;
  --lite-text-secondary: #5f6368;

  /* Elevation & Shadows */
  --lite-shadow: 0 1px 3px rgba(60, 64, 67, 0.1), 0 1px 2px rgba(60, 64, 67, 0.06);
  --lite-hover-shadow: 0 10px 20px rgba(60, 64, 67, 0.1), 0 6px 6px rgba(60, 64, 67, 0.1);

  /* Animation Rationale: cubic-bezier optimized for snappy feedback */
  --lite-ease: cubic-bezier(0.4, 0, 0.2, 1);
  --lite-transition: all 0.3s var(--lite-ease);
}

/* 1. Base Typography & Rhythm
 * --------------------------------------------------------- */
.devsite-landing-row-heading {
  letter-spacing: -0.015em;
  font-weight: 600;
  color: var(--lite-text-primary);
  margin-bottom: 12px;
  line-height: 1.25;
}

.devsite-landing-row-description {
  color: var(--lite-text-secondary);
  line-height: 1.6;
  margin-bottom: 8px; /* Further minimized spacing below descriptions */
  font-size: 16px;
}

.next-gen-hero .devsite-landing-row-description {
  font-size: 18px !important; /* Premium scaling for the marquee description */
  line-height: 1.5;
  color: #3c4043;
}

/* 2. Premium Card Treatment
 * --------------------------------------------------------- */
.devsite-landing-row-item {
  transition: var(--lite-transition);
  padding: 16px; /* Ultra-minimized internal card padding */
  border-radius: 12px;
  background: #ffffff;
  border: 1px solid var(--lite-border);
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.devsite-landing-row-item:hover {
  transform: translateY(-4px); /* Subtler lift for a more professional feel */
  box-shadow: var(--lite-hover-shadow);
  border-color: var(--lite-accent);
}

/* Universal Button Interactivity */
.devsite-landing-row-item .button,
.devsite-landing-row-item .devsite-landing-row-item-button {
  transition: var(--lite-transition) !important;
}

.devsite-landing-row-item:hover .button,
.devsite-landing-row-item:hover .devsite-landing-row-item-button {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* 3. Next-Gen Hero Styling
 * --------------------------------------------------------- */
.next-gen-hero {
  /* Dynamic radial gradient for a premium background feel */
  background: radial-gradient(circle at 70% 30%, #f1f3f4, #ffffff) !important;
  border-bottom: 1px solid var(--lite-border);
  padding: 12px 0 !important; /* Ultra-minimized hero padding */
}

.next-gen-hero .devsite-landing-row-item-media img {
  border-radius: 8px;
  max-width: 85% !important;
  box-shadow: 0 30px 60px rgba(0, 0, 0, 0.12);
  filter: saturate(1.1);
  transition: var(--lite-transition);
}

.next-gen-hero .devsite-landing-row-item:hover .devsite-landing-row-item-media img {
  transform: scale(1.02);
  filter: saturate(1.2) brightness(1.05);
}

/* 4. Action Buttons & Interactivity
 * --------------------------------------------------------- */
.primary-action-glass {
  background: var(--lite-blue) !important;
  color: #ffffff !important;
  border-radius: 8px !important;
  padding: 12px 32px !important;
  font-weight: 500 !important;
  box-shadow: 0 4px 12px var(--lite-blue-focus);
  transition: all 0.2s ease !important;
}

.primary-action-glass:hover {
  background: var(--lite-blue-hover) !important;
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(26, 115, 232, 0.32);
}

/* 5. Section-Specific Grid Optimizations
 * --------------------------------------------------------- */
.developer-pathway .devsite-landing-row-item {
  background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(4px);
}

.highlights-section .devsite-landing-row-item,
.pipeline-section .devsite-landing-row-item {
  border: 1px solid rgba(60, 64, 67, 0.08);
  box-shadow: var(--lite-shadow);
}

.highlights-section .devsite-landing-row-item-heading {
  font-size: 0.9rem; /* Reduced size to prioritize iconography */
}

.highlights-section .devsite-landing-row-item {
  text-align: center; /* Center visuals and headings for a balanced focus */
  justify-content: center; /* Vertically center in grid */
}

/* 6. Media Scaling & Hover Feedback
 * --------------------------------------------------------- */
.devsite-landing-row-item-media img,
.devsite-landing-row-item-icon img,
.devsite-landing-row-item-icon .devsite-icon {
  transition: var(--lite-transition);
}

.highlights-section .devsite-landing-row-item-media img {
  max-width: 75%;
  height: auto;
}

.pipeline-section .devsite-landing-row-item-media img {
  max-width: 40%;
  height: auto;
}

/* Ensure DevSite SVG icons can be transformed */
.devsite-landing-row-item-icon .devsite-icon {
  display: inline-block; 
}

/* Consistent hover treatment across all interactive items */
.devsite-landing-row-item:hover .devsite-landing-row-item-media img,
.devsite-landing-row-item:hover .devsite-landing-row-item-icon img,
.devsite-landing-row-item:hover .devsite-landing-row-item-icon .devsite-icon {
  transform: scale(1.1);
  filter: brightness(1.1) saturate(1.1);
}

/* 7. High-Impact Finale (CTA)
 * --------------------------------------------------------- */
.next-gen-cta {
  background: #f1f3f4 !important;
  padding: 20px 0 !important; /* Ultra-minimized CTA padding */
  box-shadow: inset 0 1px 0 var(--lite-border); /* Subtle delimiter */
}

/* Enhancing 'Grey' sections with a subtle depth */
.devsite-landing-row[background="grey"] {
  box-shadow: inset 0 8px 12px -10px rgba(0,0,0,0.05), inset 0 -8px 12px -10px rgba(0,0,0,0.05);
}

.next-gen-cta .devsite-landing-row-heading,
.next-gen-cta .devsite-landing-row-description {
  color: #202124 !important; /* High-readability black text on light grey surface */
}

.next-gen-cta .button.primary {
  background: var(--lite-blue) !important;
  color: #ffffff !important;
  font-weight: 600;
}

/* 8. Mobile & Tablet Responsiveness 
 * --------------------------------------------------------- */
@media screen and (max-width: 768px) {
  .devsite-landing-row-heading {
    font-size: 1.75rem !important;
  }

  .devsite-landing-row-description {
    font-size: 15px;
  }

  .devsite-landing-row-item {
    padding: 16px; /* Optimized mobile padding */
  }

  .highlights-section .devsite-landing-row-item-media img {
    max-width: 40% !important;
  }

  .pipeline-section .devsite-landing-row-item-media img {
    max-width: 30% !important;
  }
}

@media screen and (max-width: 480px) {
  .next-gen-hero {
    padding: 40px 0 !important;
  }
  
  .next-gen-cta {
    padding: 40px 20px !important;
    text-align: center;
  }
}

      </style>
  
  
  
    <div class="devsite-banner devsite-banner-announcement nocontent" data-nosnippet
      
        
    background="google-blue"
  
      >
      <div class="devsite-banner-message">
        <div class="devsite-banner-message-text">
          <a href="https://ai.google.dev/edge/ai-edge-portal"><b>Introducing Google AI Edge Portal</b></a>: Benchmark Edge AI at scale. <a href="https://docs.google.com/forms/d/e/1FAIpQLSfTcGPycQve8TLAsfH46pBlXBZe9FrgJAClwbF7DeL1LgVn4Q/viewform">Sign-up</a> to request access during private preview.
        </div>
      </div>
    </div>
  
  
  

  <div class="devsite-article-meta nocontent" role="navigation" data-nosnippet>
    
    
    <ul class="devsite-breadcrumb-list"
  
    aria-label="Breadcrumb">
  
  <li class="devsite-breadcrumb-item
             ">
    
    
    
      
        
  <a href="https://developers.google.com/"
      
        class="devsite-breadcrumb-link gc-analytics-event"
      
        data-category="Site-Wide Custom Events"
      
        data-label="Breadcrumbs"
      
        data-value="1"
      
        track-type="globalNav"
      
        track-name="breadcrumb"
      
        track-metadata-position="1"
      
        track-metadata-eventdetail=""
      
    >
    
          Home
        
  </a>
  
      
    
  </li>
  
  <li class="devsite-breadcrumb-item
             ">
    
      
      <div class="devsite-breadcrumb-guillemet material-icons" aria-hidden="true"></div>
    
    
    
      
        
  <a href="https://developers.google.com/products"
      
        class="devsite-breadcrumb-link gc-analytics-event"
      
        data-category="Site-Wide Custom Events"
      
        data-label="Breadcrumbs"
      
        data-value="2"
      
        track-type="globalNav"
      
        track-name="breadcrumb"
      
        track-metadata-position="2"
      
        track-metadata-eventdetail=""
      
    >
    
          Products
        
  </a>
  
      
    
  </li>
  
  <li class="devsite-breadcrumb-item
             ">
    
      
      <div class="devsite-breadcrumb-guillemet material-icons" aria-hidden="true"></div>
    
    
    
      
        
  <a href="https://developers.google.com/edge"
      
        class="devsite-breadcrumb-link gc-analytics-event"
      
        data-category="Site-Wide Custom Events"
      
        data-label="Breadcrumbs"
      
        data-value="3"
      
        track-type="globalNav"
      
        track-name="breadcrumb"
      
        track-metadata-position="3"
      
        track-metadata-eventdetail="Google AI Edge"
      
    >
    
          Google AI Edge
        
  </a>
  
      
    
  </li>
  
  <li class="devsite-breadcrumb-item
             ">
    
      
      <div class="devsite-breadcrumb-guillemet material-icons" aria-hidden="true"></div>
    
    
    
      
        
  <a href="https://developers.google.com/edge/litert"
      
        class="devsite-breadcrumb-link gc-analytics-event"
      
        data-category="Site-Wide Custom Events"
      
        data-label="Breadcrumbs"
      
        data-value="4"
      
        track-type="globalNav"
      
        track-name="breadcrumb"
      
        track-metadata-position="4"
      
        track-metadata-eventdetail=""
      
    >
    
          LiteRT
        
  </a>
  
      
    
  </li>
  
</ul>
    
  </div>
  
    <devsite-feedback
  position="header"
  project-name="Google AI Edge"
  product-id="5336252"
  bucket="documentation"
  context=""
  version="t-devsite-webserver-20260630-r00-rc00.478639290528629295"
  data-label="Send Feedback Button"
  track-type="feedback"
  track-name="sendFeedbackLink"
  track-metadata-position="header"
  class="nocontent"
  data-nosnippet
  
  
  
    project-icon="https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers/images/touchicon-180-new.png"
  
  
  
  >

  <button>
  
    
    Send feedback
  
  </button>
</devsite-feedback>
  <devsite-actions hidden data-nosnippet><devsite-feature-tooltip
      ack-key="AckCollectionsBookmarkTooltipDismiss"
      analytics-category="Site-Wide Custom Events"
      analytics-action-show="Callout Profile displayed"
      analytics-action-close="Callout Profile dismissed"
      analytics-label="Create Collection Callout"
      class="devsite-page-bookmark-tooltip nocontent"
      data-nosnippet
      dismiss-button="true"
      id="devsite-collections-dropdown"
      
      dismiss-button-text="Dismiss"

      
      close-button-text="Got it">

    
    
      <devsite-bookmark></devsite-bookmark>
    

    <span slot="popout-heading">
      
      Stay organized with collections
    </span>
    <span slot="popout-contents">
      
      Save and categorize content based on your preferences.
    </span>
  </devsite-feature-tooltip></devsite-actions>
  
    
  

  <devsite-toc class="devsite-nav"
    depth="2"
    devsite-toc-embedded
    disabled>
  </devsite-toc>
  <div class="devsite-article-body clearfix
  ">

  
    
  <section class="devsite-landing-row devsite-landing-row-1-up devsite-landing-row-50 devsite-landing-row-marquee devsite-landing-row-no-image-background devsite-landing-row-padding-small next-gen-hero"
           
           
           
    header-position="top"
  >
    <div class="devsite-landing-row-inner">

    
      

      
      

      

        <div class="devsite-landing-row-group">
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/logo-litert.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/logo-litert_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/logo-litert_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/logo-litert_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/logo-litert_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/logo-litert_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/logo-litert_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/logo-litert_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/logo-litert_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/logo-litert_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/logo-litert_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/logo-litert_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 100vw, (max-width: 840px) 50vw, 708px"
         
         
         fetchpriority="high">
  </picture>
  
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="litert-is-googles-on-device-framework-for-high-performance-ml-genai-deployment-on-edge-platforms"
        data-text="LiteRT is Google's on-device framework for high-performance ML & GenAI deployment on edge platforms."
        class="hide-from-toc no-link"
        tabindex="0">
      
    
        LiteRT is Google's on-device framework for high-performance ML & GenAI deployment on edge platforms.
      
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Efficient conversion, runtime, and optimization for on-device machine learning.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
        </div>
      

    
    </div>
  </section>

  <section class="devsite-landing-row devsite-landing-row-1-up devsite-landing-row-75 devsite-landing-row-no-image-background devsite-landing-row-padding-small"
           
    background="grey"
  
           
           
    header-position="top"
  >
    <div class="devsite-landing-row-inner">

    
      

      
      

      

        <div class="devsite-landing-row-group">
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
    
  
  <picture>
    
    <img alt=""
         
         src=""
         srcset=""
         
         
         sizes="(max-width: 600px) 100vw, (max-width: 840px) 50vw, 342px"
         
         loading="lazy"
         >
  </picture>
  
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="built-on-the-battle-tested-foundation-of-tensorflow-lite"
        data-text="Built on the battle-tested foundation of TensorFlow Lite"
        class="hide-from-toc no-link"
        tabindex="0">
      
    
        Built on the battle-tested foundation of TensorFlow Lite
      
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            LiteRT isn't just new; it's the next generation of the world's most widely deployed machine learning runtime. It powers the apps you use every day, delivering low latency and high privacy on billions of devices.

          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
        </div>
      

    
    </div>
  </section>

  <section class="devsite-landing-row devsite-landing-row-4-up devsite-landing-row-header-centered devsite-landing-row-logos devsite-landing-row-padding-small"
           
    background="white"
  
           
           
    header-position="top"
  >
    <div class="devsite-landing-row-inner">

    
      
      <header class="devsite-landing-row-header"
              >

        

        
        <div class="devsite-landing-row-header-text">

          
    <h2 id="trusted-by-the-most-critical-google-apps"
        data-text="Trusted by the most critical Google apps"
        
        tabindex="0">
      
    
        Trusted by the most critical Google apps
      
  
    </h2>
  

          
            <div class="devsite-landing-row-description">
              100K+ applications, billions of global users
            </div>
          
        </div>
        

        
      </header>
      

      
      

      

        <div class="devsite-landing-row-group">
        
          <div class="devsite-landing-row-item devsite-landing-row-item-no-description"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-custom-image"
        
        >
  <div class="devsite-landing-row-item-custom-image-icon-wrapper">
  
    
    
  
    
  <div class="devsite-landing-row-item-custom-image-icon-container"
       
       
       
    size="medium"
  >
  
    <picture>
      
      <img class="devsite-landing-row-item-custom-image-icon"
           alt=""
           src="https://developers.google.com/static/edge/litert/images/landing/logo-gmail.svg"
           srcset="https://developers.google.com/static/edge/litert/images/landing/logo-gmail.svg"
           sizes="192px"
           loading="lazy"
           >
    </picture>
  
  </div>
  
  

  
  
  </div>
</figure>
  
</div>


    
  

</div>
        
          <div class="devsite-landing-row-item devsite-landing-row-item-no-description"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-custom-image"
        
        >
  <div class="devsite-landing-row-item-custom-image-icon-wrapper">
  
    
    
  
    
  <div class="devsite-landing-row-item-custom-image-icon-container"
       
       
       
    size="medium"
  >
  
    <picture>
      
      <img class="devsite-landing-row-item-custom-image-icon"
           alt=""
           src="https://developers.google.com/static/edge/litert/images/landing/logo-yt.svg"
           srcset="https://developers.google.com/static/edge/litert/images/landing/logo-yt.svg"
           sizes="192px"
           loading="lazy"
           >
    </picture>
  
  </div>
  
  

  
  
  </div>
</figure>
  
</div>


    
  

</div>
        
          <div class="devsite-landing-row-item devsite-landing-row-item-no-description"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-custom-image"
        
        >
  <div class="devsite-landing-row-item-custom-image-icon-wrapper">
  
    
    
  
    
  <div class="devsite-landing-row-item-custom-image-icon-container"
       
       
       
    size="medium"
  >
  
    <picture>
      
      <img class="devsite-landing-row-item-custom-image-icon"
           alt=""
           src="https://developers.google.com/static/edge/litert/images/landing/logo-maps.svg"
           srcset="https://developers.google.com/static/edge/litert/images/landing/logo-maps.svg"
           sizes="192px"
           loading="lazy"
           >
    </picture>
  
  </div>
  
  

  
  
  </div>
</figure>
  
</div>


    
  

</div>
        
          <div class="devsite-landing-row-item devsite-landing-row-item-no-description"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-custom-image"
        
        >
  <div class="devsite-landing-row-item-custom-image-icon-wrapper">
  
    
    
  
    
  <div class="devsite-landing-row-item-custom-image-icon-container"
       
       
       
    size="medium"
  >
  
    <picture>
      
      <img class="devsite-landing-row-item-custom-image-icon"
           alt=""
           src="https://developers.google.com/static/edge/litert/images/landing/logo-photos.svg"
           srcset="https://developers.google.com/static/edge/litert/images/landing/logo-photos.svg"
           sizes="192px"
           loading="lazy"
           >
    </picture>
  
  </div>
  
  

  
  
  </div>
</figure>
  
</div>


    
  

</div>
        
        </div>
      

    
    </div>
  </section>

  <section class="devsite-landing-row devsite-landing-row-4-up devsite-landing-row-cards devsite-landing-row-padding-small devsite-landing-row-no-image-background highlights-section"
           
    background="grey"
  
           
           
    header-position="top"
  >
    <div class="devsite-landing-row-inner">

    
      
      <header class="devsite-landing-row-header"
              >

        

        
        <div class="devsite-landing-row-header-text">

          
    <h2 id="litert-highlights"
        data-text="LiteRT Highlights"
        
        tabindex="0">
      
    
        LiteRT Highlights
      
  
    </h2>
  

          
        </div>
        

        
      </header>
      

      
      

      

        <div class="devsite-landing-row-group">
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/crossplatform.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/crossplatform_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/crossplatform_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/crossplatform_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/crossplatform_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/crossplatform_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/crossplatform_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/crossplatform_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/crossplatform_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/crossplatform_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/crossplatform_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/crossplatform_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 50vw, (max-width: 840px) 25vw, 342px"
         
         loading="lazy"
         >
  </picture>
  
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="cross-platform-ready"
        data-text="Cross Platform Ready"
        class="hide-from-toc no-link"
        tabindex="0">
      
    
        Cross Platform Ready
      
  
    </h3>
  

        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/genAI.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/genAI_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/genAI_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/genAI_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/genAI_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/genAI_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/genAI_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/genAI_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/genAI_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/genAI_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/genAI_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/genAI_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 50vw, (max-width: 840px) 25vw, 342px"
         
         loading="lazy"
         >
  </picture>
  
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="unleash-genai"
        data-text="Unleash GenAI"
        class="hide-from-toc no-link"
        tabindex="0">
      
    
        Unleash GenAI
      
  
    </h3>
  

        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/hardware.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/hardware_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/hardware_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/hardware_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/hardware_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/hardware_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/hardware_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/hardware_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/hardware_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/hardware_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/hardware_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/hardware_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 50vw, (max-width: 840px) 25vw, 342px"
         
         loading="lazy"
         >
  </picture>
  
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="simplified-hardware-acceleration"
        data-text="Simplified hardware acceleration"
        class="hide-from-toc no-link"
        tabindex="0">
      
    
        Simplified hardware acceleration
      
  
    </h3>
  

        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/multiframework.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/multiframework_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/multiframework_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/multiframework_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/multiframework_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/multiframework_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/multiframework_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/multiframework_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/multiframework_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/multiframework_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/multiframework_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/multiframework_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 50vw, (max-width: 840px) 25vw, 342px"
         
         loading="lazy"
         >
  </picture>
  
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="multi-framework-support"
        data-text="Multi-framework support"
        class="hide-from-toc no-link"
        tabindex="0">
      
    
        Multi-framework support
      
  
    </h3>
  

        

        

        
      </div>
    </div>
    
  

</div>
        
        </div>
      

    
    </div>
  </section>

  <section class="devsite-landing-row devsite-landing-row-3-up devsite-landing-row-cards devsite-landing-row-header-centered devsite-landing-row-padding-small devsite-landing-row-no-image-background pipeline-section"
           
    background="white"
  
           
           
    header-position="top"
  >
    <div class="devsite-landing-row-inner">

    
      
      <header class="devsite-landing-row-header"
              >

        

        
        <div class="devsite-landing-row-header-text">

          
    <h2 id="deploy-via-litert"
        data-text="Deploy via LiteRT"
        
        tabindex="0">
      
    
        Deploy via LiteRT
      
  
    </h2>
  

          
            <div class="devsite-landing-row-description">
              Streamline your deep learning workflow from training to on-device deployment.
            </div>
          
        </div>
        

        
      </header>
      

      
      

      

        <div class="devsite-landing-row-group">
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
  <a href="https://developers.google.com/edge/litert/conversion/overview">
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/obtain-model.svg"
         srcset="https://developers.google.com/static/edge/litert/images/landing/obtain-model.svg"
         
         
         sizes="(max-width: 840px) 50vw, 464px"
         
         loading="lazy"
         >
  </picture>
  
  </a>
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="1obtain-a-model"
        data-text="1.Obtain a model"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.google.com/edge/litert/conversion/overview">
    
        1.Obtain a model
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Use .tflite pre-trained models or convert PyTorch, JAX or TensorFlow models to .tflite.
          </div>
        

        

        
          <div class="devsite-landing-row-item-buttons">
  

  
  <a href="https://developers.google.com/edge/litert/conversion/overview"
  
    
      class="button
             button-white
             "
    
    
    >Learn about conversion</a>

</div>
        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
  <a href="https://developers.google.com/edge/litert/conversion/tensorflow/quantization/model_optimization">
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/optimize-model.svg"
         srcset="https://developers.google.com/static/edge/litert/images/landing/optimize-model.svg"
         
         
         sizes="(max-width: 840px) 50vw, 464px"
         
         loading="lazy"
         >
  </picture>
  
  </a>
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="2optimize"
        data-text="2.Optimize"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.google.com/edge/litert/conversion/tensorflow/quantization/model_optimization">
    
        2.Optimize
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Use the LiteRT optimization toolkit to quantize your models post-training.
          </div>
        

        

        
          <div class="devsite-landing-row-item-buttons">
  

  
  <a href="https://developers.google.com/edge/litert/conversion/tensorflow/quantization/model_optimization"
  
    
      class="button
             button-white
             "
    
    
    >Explore optimization</a>

</div>
        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
  <a href="https://developers.google.com/edge/litert/overview#hardware-acceleration">
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/run-model.svg"
         srcset="https://developers.google.com/static/edge/litert/images/landing/run-model.svg"
         
         
         sizes="(max-width: 840px) 50vw, 464px"
         
         loading="lazy"
         >
  </picture>
  
  </a>
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="3run"
        data-text="3.Run"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.google.com/edge/litert/overview#hardware-acceleration">
    
        3.Run
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Deploy your model with LiteRT and pick the optimal accelerator for your app.
          </div>
        

        

        
          <div class="devsite-landing-row-item-buttons">
  

  
  <a href="https://developers.google.com/edge/litert/overview#hardware-acceleration"
  
    
      class="button
             button-white
             "
    
    
    >View deployment targets</a>

</div>
        
      </div>
    </div>
    
  

</div>
        
        </div>
      

    
    </div>
  </section>

  <section class="devsite-landing-row devsite-landing-row-2-up devsite-landing-row-cards devsite-landing-row-header-centered devsite-landing-row-padding-small developer-pathway"
           
    background="grey"
  
           
           
    header-position="top"
  >
    <div class="devsite-landing-row-inner">

    
      
      <header class="devsite-landing-row-header"
              >

        

        
        <div class="devsite-landing-row-header-text">

          
    <h2 id="choose-your-development-path"
        data-text="Choose Your Development Path"
        
        tabindex="0">
      
    
        Choose Your Development Path
      
  
    </h2>
  

          
            <div class="devsite-landing-row-description">
              Use LiteRT to deploy AI anywhere—from high-performance mobile apps to resource-constrained IoT devices.
            </div>
          
        </div>
        

        
      </header>
      

      
      

      

        <div class="devsite-landing-row-group">
        
          <div class="devsite-landing-row-item devsite-landing-row-item-no-media"
     
     
     
    description-position="bottom"
  >

  
    

    
    <div class="devsite-landing-row-item-description"
         
    icon-position="left"
  >

      
  
  <a href="https://developers.google.com/edge/litert/migration">
    
  <div class="devsite-landing-row-item-icon-container"
       
       
       
    size="medium"
  >
  
    <div class="devsite-landing-row-item-icon material-icons" aria-hidden="true">
      sync
    </div>
  
  </div>
  
  </a>
  


      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="existing-tflite-user"
        data-text="Existing TFLite User"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.google.com/edge/litert/migration">
    
        Existing TFLite User
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Transitioning to LiteRT to leverage enhanced performance and unified APIs across platforms (Android, Desktop, Web).
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item devsite-landing-row-item-no-media"
     
     
     
    description-position="bottom"
  >

  
    

    
    <div class="devsite-landing-row-item-description"
         
    icon-position="left"
  >

      
  
  <a href="https://developers.google.com/edge/litert/conversion/overview">
    
  <div class="devsite-landing-row-item-icon-container"
       
       
       
    size="medium"
  >
  
    <div class="devsite-landing-row-item-icon material-icons" aria-hidden="true">
      camera
    </div>
  
  </div>
  
  </a>
  


      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="byom-bring-your-own-models"
        data-text="BYOM : Bring your own Models"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.google.com/edge/litert/conversion/overview">
    
        BYOM : Bring your own Models
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Have a PyTorch model, looking to implement on-device vision or audio experiences.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item devsite-landing-row-item-no-media"
     
     
     
    description-position="bottom"
  >

  
    

    
    <div class="devsite-landing-row-item-description"
         
    icon-position="left"
  >

      
  
  <a href="https://developers.google.com/edge/litert/next/litert_lm_npu">
    
  <div class="devsite-landing-row-item-icon-container"
       
       
       
    size="medium"
  >
  
    <div class="devsite-landing-row-item-icon material-icons" aria-hidden="true">
      auto_awesome
    </div>
  
  </div>
  
  </a>
  


      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="deploying-generative-ai-models"
        data-text="Deploying Generative AI Models"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.google.com/edge/litert/next/litert_lm_npu">
    
        Deploying Generative AI Models
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Creating sophisticated on-device chatbots using optimized open-weight GenAI models like Gemma or another open-weight model.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item devsite-landing-row-item-no-media"
     
     
     
    description-position="bottom"
  >

  
    

    
    <div class="devsite-landing-row-item-description"
         
    icon-position="left"
  >

      
  
  <a href="https://developers.google.com/edge/litert/genai/overview">
    
  <div class="devsite-landing-row-item-icon-container"
       
       
       
    size="medium"
  >
  
    <div class="devsite-landing-row-item-icon material-icons" aria-hidden="true">
      developer_board
    </div>
  
  </div>
  
  </a>
  


      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="advanced-model-expert"
        data-text="[Advanced] Model Expert"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.google.com/edge/litert/genai/overview">
    
        [Advanced] Model Expert
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Authoring custom models or performing deep hardware-specific CPU/GPU/NPU optimizations for peak performance.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
        </div>
      

    
    </div>
  </section>

  <section class="devsite-landing-row devsite-landing-row-3-up devsite-landing-row-cards devsite-landing-row-padding-small pipeline-section"
           
    background="white"
  
           
           
    header-position="top"
  >
    <div class="devsite-landing-row-inner">

    
      
      <header class="devsite-landing-row-header"
              >

        

        
        <div class="devsite-landing-row-header-text">

          
    <h2 id="samples-models-and-demo"
        data-text="Samples, models, and demo"
        
        tabindex="0">
      
    
        Samples, models, and demo
      
  
    </h2>
  

          
        </div>
        

        
      </header>
      

      
      

      

        <div class="devsite-landing-row-group">
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-custom-image"
        
    background="white"
  
        >
  <div class="devsite-landing-row-item-custom-image-icon-wrapper">
  
  <a href="https://github.com/google-ai-edge/litert-samples/tree/main/compiled_model_api">
    
    
  
    
  <div class="devsite-landing-row-item-custom-image-icon-container"
       
    background="white"
  
       
       
    size="medium"
  >
  
    <picture>
      
      <img class="devsite-landing-row-item-custom-image-icon"
           alt=""
           src="https://developers.google.com/static/edge/litert/images/landing/github.svg"
           srcset="https://developers.google.com/static/edge/litert/images/landing/github.svg"
           sizes="192px"
           loading="lazy"
           >
    </picture>
  
  </div>
  
  

  
  </a>
  
  </div>
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="see-litert-sample-app-on-github"
        data-text="See LiteRT sample app on GitHub"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://github.com/google-ai-edge/litert-samples/tree/main/compiled_model_api">
    
        See LiteRT sample app on GitHub
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Complete, end-to-end sample apps.

          </div>
        

        

        
          <div class="devsite-landing-row-item-buttons">
  

  
  <a href="https://github.com/google-ai-edge/litert-samples/tree/main/compiled_model_api"
  
    class="button
      "
    
    
    >See sample app</a>

</div>
        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-custom-image"
        
    background="white"
  
        >
  <div class="devsite-landing-row-item-custom-image-icon-wrapper">
  
  <a href="https://huggingface.co/litert-community">
    
    
  
    
  <div class="devsite-landing-row-item-custom-image-icon-container"
       
    background="white"
  
       
       
    size="medium"
  >
  
    <picture>
      
      <img class="devsite-landing-row-item-custom-image-icon"
           alt=""
           src="https://developers.google.com/static/edge/litert/images/landing/genai.svg"
           srcset="https://developers.google.com/static/edge/litert/images/landing/genai.svg"
           sizes="192px"
           loading="lazy"
           >
    </picture>
  
  </div>
  
  

  
  </a>
  
  </div>
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="see-genai-models"
        data-text="See genAI models"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://huggingface.co/litert-community">
    
        See genAI models
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Pre-trained, out-of-the-box Gen AI models.

          </div>
        

        

        
          <div class="devsite-landing-row-item-buttons">
  

  
  <a href="https://huggingface.co/litert-community"
  
    class="button
      "
    
    
    >Go to HuggingFace</a>

</div>
        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-custom-image"
        
    background="white"
  
        >
  <div class="devsite-landing-row-item-custom-image-icon-wrapper">
  
  <a href="https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery">
    
    
  
    
  <div class="devsite-landing-row-item-custom-image-icon-container"
       
    background="white"
  
       
       
    size="medium"
  >
  
    <picture>
      
      <img class="devsite-landing-row-item-custom-image-icon"
           alt=""
           src="https://developers.google.com/static/edge/litert/images/landing/Gallery.png"
           srcset="https://developers.google.com/static/edge/litert/images/landing/Gallery_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/Gallery_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/Gallery_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/Gallery_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/Gallery_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/Gallery_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/Gallery_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/Gallery_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/Gallery_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/Gallery_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/Gallery_2880.png 2880w"
           sizes="192px"
           loading="lazy"
           >
    </picture>
  
  </div>
  
  

  
  </a>
  
  </div>
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="see-demos-google-ai-edge-gallery-app"
        data-text="See demos - Google AI Edge Gallery App"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery">
    
        See demos - Google AI Edge Gallery App
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            A gallery that showcases on-device ML/GenAI use cases using LiteRT.

          </div>
        

        

        
          <div class="devsite-landing-row-item-buttons">
  

  
  <a href="https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery"
  
    class="button
      "
    
    
    >Open Play Store</a>

</div>
        
      </div>
    </div>
    
  

</div>
        
        </div>
      

    
    </div>
  </section>

  <section class="devsite-landing-row devsite-landing-row-2-up devsite-landing-row-cards devsite-landing-row-padding-small"
           
    background="grey"
  
           
           
    header-position="top"
  >
    <div class="devsite-landing-row-inner">

    
      
      <header class="devsite-landing-row-header"
              >

        

        
        <div class="devsite-landing-row-header-text">

          
    <h2 id="blogs-and-announcements"
        data-text="Blogs and Announcements"
        
        tabindex="0">
      
    
        Blogs and Announcements
      
  
    </h2>
  

          
            <div class="devsite-landing-row-description">
              Stay up to date with the latest announcements, technical deep dives, and performance benchmarks from the LiteRT team.
            </div>
          
        </div>
        

        
          <div class="devsite-landing-row-header-buttons">
  

  
  <a href="https://developers.google.com/edge/litert/overview"
  
    class="button primary
      "
    
    
    >Explore more blogs</a>

</div>
        
      </header>
      

      
      

      

        <div class="devsite-landing-row-group">
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
  <a href="https://developers.googleblog.com/building-real-world-on-device-ai-with-litert-and-npu/">
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/BlogNPU.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/BlogNPU_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/BlogNPU_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/BlogNPU_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/BlogNPU_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/BlogNPU_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/BlogNPU_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/BlogNPU_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/BlogNPU_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/BlogNPU_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/BlogNPU_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/BlogNPU_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 100vw, (max-width: 840px) 50vw, 708px"
         
         loading="lazy"
         >
  </picture>
  
  </a>
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="building-real-world-on-device-ai-with-litert-and-npu"
        data-text="Building real-world on-device AI with LiteRT and NPU"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.googleblog.com/building-real-world-on-device-ai-with-litert-and-npu/">
    
        Building real-world on-device AI with LiteRT and NPU
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Learn how industry leaders build real-world, high-performance on-device AI applications using LiteRT and NPUs.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
  <a href="https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/">
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/gemma4_banner_2_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 100vw, (max-width: 840px) 50vw, 708px"
         
         loading="lazy"
         >
  </picture>
  
  </a>
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4"
        data-text="Bring state-of-the-art agentic skills to the edge with Gemma 4"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/">
    
        Bring state-of-the-art agentic skills to the edge with Gemma 4
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Deploy agentic and multi-step planning capabilities entirely on-device with the new Gemma 4 family and LiteRT.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
  <a href="https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/">
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/universal-framework.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/universal-framework_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/universal-framework_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/universal-framework_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/universal-framework_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/universal-framework_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/universal-framework_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/universal-framework_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/universal-framework_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/universal-framework_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/universal-framework_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/universal-framework_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 100vw, (max-width: 840px) 50vw, 708px"
         
         loading="lazy"
         >
  </picture>
  
  </a>
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="litert-the-universal-framework-for-on-device-ai"
        data-text="LiteRT: The universal framework for on-device AI"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/">
    
        LiteRT: The universal framework for on-device AI
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Google's unified on-device ML framework, evolving from TFLite for high-performance deployment.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
  <a href="https://developers.googleblog.com/mediatek-npu-and-litert-powering-the-next-generation-of-on-device-ai/">
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/blog.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/blog_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/blog_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/blog_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/blog_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/blog_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/blog_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/blog_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/blog_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/blog_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/blog_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/blog_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 100vw, (max-width: 840px) 50vw, 708px"
         
         loading="lazy"
         >
  </picture>
  
  </a>
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="mediatek-npu-and-litert-powering-the-next-generation-of-on-device-ai"
        data-text="MediaTek NPU and LiteRT: Powering the next generation of on-device AI"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.googleblog.com/mediatek-npu-and-litert-powering-the-next-generation-of-on-device-ai/">
    
        MediaTek NPU and LiteRT: Powering the next generation of on-device AI
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Expanding NPU acceleration support to MediaTek chipsets for high-efficiency AI.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
  <a href="https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/">
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/blog.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/blog_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/blog_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/blog_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/blog_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/blog_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/blog_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/blog_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/blog_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/blog_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/blog_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/blog_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 100vw, (max-width: 840px) 50vw, 708px"
         
         loading="lazy"
         >
  </picture>
  
  </a>
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="unlocking-peak-performance-on-qualcomm-npu-with-litert"
        data-text="Unlocking Peak Performance on Qualcomm NPU with LiteRT"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/">
    
        Unlocking Peak Performance on Qualcomm NPU with LiteRT
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Unlocking breakthrough performance for generative AI on Qualcomm Neural Processing Units.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
  <a href="https://developers.googleblog.com/litert-maximum-performance-simplified/">
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/banner-litert-google-io_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 100vw, (max-width: 840px) 50vw, 708px"
         
         loading="lazy"
         >
  </picture>
  
  </a>
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="litert-maximum-performance-simplified"
        data-text="LiteRT: Maximum Performance, Simplified"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.googleblog.com/litert-maximum-performance-simplified/">
    
        LiteRT: Maximum Performance, Simplified
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Introducing the CompiledModel API for automated hardware selection and async execution.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
  <a href="https://developers.googleblog.com/on-device-genai-in-chrome-chromebook-plus-and-pixel-watch-with-litert-lm/">
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/blog.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/blog_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/blog_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/blog_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/blog_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/blog_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/blog_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/blog_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/blog_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/blog_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/blog_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/blog_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 100vw, (max-width: 840px) 50vw, 708px"
         
         loading="lazy"
         >
  </picture>
  
  </a>
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="on-device-genai-in-chrome-chromebook-plus-and-pixel-watch-with-litert-lm"
        data-text="On-device GenAI in Chrome, Chromebook Plus, and Pixel Watch with LiteRT-LM"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.googleblog.com/on-device-genai-in-chrome-chromebook-plus-and-pixel-watch-with-litert-lm/">
    
        On-device GenAI in Chrome, Chromebook Plus, and Pixel Watch with LiteRT-LM
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Deploy language models on wearables and browser-based platforms using LiteRT-LM.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item"
     
     
     
    description-position="bottom"
  >

  
    
<div class="devsite-landing-row-item-media
            ">
  
    <figure class="devsite-landing-row-item-image">
  
  <a href="https://developers.googleblog.com/google-ai-edge-small-language-models-multimodality-rag-function-calling/">
    
  
  <picture>
    
    <img alt=""
         
         src="https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original.png"
         srcset="https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original_36.png 36w,https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original_48.png 48w,https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original_72.png 72w,https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original_96.png 96w,https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original_480.png 480w,https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original_720.png 720w,https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original_856.png 856w,https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original_960.png 960w,https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original_1440.png 1440w,https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original_1920.png 1920w,https://developers.google.com/static/edge/litert/images/landing/O25-BHero-AI-4.original_2880.png 2880w"
         
         
         sizes="(max-width: 600px) 100vw, (max-width: 840px) 50vw, 708px"
         
         loading="lazy"
         >
  </picture>
  
  </a>
  
</figure>
  
</div>


    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="google-ai-edge-small-language-models-multimodality-and-function-calling"
        data-text="Google AI Edge small language models, multimodality, and function calling"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.googleblog.com/google-ai-edge-small-language-models-multimodality-rag-function-calling/">
    
        Google AI Edge small language models, multimodality, and function calling
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Latest insights on RAG, multimodality, and function calling for edge language models
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
        </div>
      

    
    </div>
  </section>

  <section class="devsite-landing-row devsite-landing-row-2-up devsite-landing-row-cards devsite-landing-row-padding-small"
           
    background="white"
  
           
           
    header-position="top"
  >
    <div class="devsite-landing-row-inner">

    
      
      <header class="devsite-landing-row-header"
              >

        

        
        <div class="devsite-landing-row-header-text">

          
    <h2 id="join-the-community"
        data-text="Join the Community"
        
        tabindex="0">
      
    
        Join the Community
      
  
    </h2>
  

          
        </div>
        

        
      </header>
      

      
      

      

        <div class="devsite-landing-row-group">
        
          <div class="devsite-landing-row-item devsite-landing-row-item-no-media"
     
     
     
    description-position="bottom"
  >

  
    

    
    <div class="devsite-landing-row-item-description"
         
    icon-position="left"
  >

      
  
  <a href="https://github.com/google-ai-edge/LiteRT">
    
  <div class="devsite-landing-row-item-icon-container"
       
       
       
    size="medium"
  >
  
    <picture>
      
      <img class="devsite-landing-row-item-icon"
           alt=""
           src="https://developers.google.com/static/edge/litert/images/landing/github.svg"
           srcset="https://developers.google.com/static/edge/litert/images/landing/github.svg"
           sizes="64px"
           loading="lazy"
           >
    </picture>
  
  </div>
  
  </a>
  


      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="litert-github-community"
        data-text="LiteRT GitHub Community"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://github.com/google-ai-edge/LiteRT">
    
        LiteRT GitHub Community
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Contribute directly to the project and collaborate with core developers.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
          <div class="devsite-landing-row-item devsite-landing-row-item-no-media"
     
     
     
    description-position="bottom"
  >

  
    

    
    <div class="devsite-landing-row-item-description"
         
    icon-position="left"
  >

      
  
  <a href="https://huggingface.co/litert-community">
    
  <div class="devsite-landing-row-item-icon-container"
       
       
       
    size="medium"
  >
  
    <picture>
      
      <img class="devsite-landing-row-item-icon"
           alt=""
           src="https://developers.google.com/static/edge/litert/images/landing/huggingface-color.svg"
           srcset="https://developers.google.com/static/edge/litert/images/landing/huggingface-color.svg"
           sizes="64px"
           loading="lazy"
           >
    </picture>
  
  </div>
  
  </a>
  


      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="hugging-face-hub"
        data-text="Hugging Face Hub"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://huggingface.co/litert-community">
    
        Hugging Face Hub
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Access optimized open-weight models on the Hugging Face Hub.
          </div>
        

        

        
      </div>
    </div>
    
  

</div>
        
        </div>
      

    
    </div>
  </section>

  <section class="devsite-landing-row devsite-landing-row-1-up devsite-landing-row-cta devsite-landing-row-50 devsite-landing-row-padding-small next-gen-cta"
           
    background="grey"
  
           
           
    header-position="top"
  >
    <div class="devsite-landing-row-inner">

    
      

      
      

      

        <div class="devsite-landing-row-group">
        
          <div class="devsite-landing-row-item devsite-landing-row-item-no-media"
     
     
     
    description-position="bottom"
  >

  
    

    
    <div class="devsite-landing-row-item-description"
         >

      

      <div class="devsite-landing-row-item-body">
        

        
    <h3 id="start-your-litert-journey"
        data-text="Start Your LiteRT Journey"
        class="hide-from-toc no-link"
        tabindex="0">
      
  <a href="https://developers.google.com/edge/litert/overview">
    
        Start Your LiteRT Journey
      
  </a>
  
    </h3>
  

        
          <div class="devsite-landing-row-item-description-content">
            Ready to take your on-device ML to the next level? Explore the documentation and start building today.
          </div>
        

        

        
          <div class="devsite-landing-row-item-buttons">
  

  
  <a href="https://developers.google.com/edge/litert/overview"
  
    class="button primary
      "
    
    
    >Explore the Docs</a>

</div>
        
      </div>
    </div>
    
  

</div>
        
        </div>
      

    
    </div>
  </section>

  

  
</div>

  

  <div class="devsite-floating-action-buttons"></div></article>


<devsite-content-footer class="nocontent" data-nosnippet>
  <p>Except as otherwise noted, the content of this page is licensed under the <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 License</a>, and code samples are licensed under the <a href="https://www.apache.org/licenses/LICENSE-2.0">Apache 2.0 License</a>. For details, see the <a href="https://developers.google.com/site-policies">Google Developers Site Policies</a>. Java is a registered trademark of Oracle and/or its affiliates.</p>
  <p>Last updated 2026-05-28 UTC.</p>
</devsite-content-footer>


<devsite-notification
>
</devsite-notification>


  
<div class="devsite-content-data">
  
    
    
    <template class="devsite-thumb-rating-feedback">
      <devsite-feedback
  position="thumb-rating"
  project-name="Google AI Edge"
  product-id="5336252"
  bucket="documentation"
  context=""
  version="t-devsite-webserver-20260630-r00-rc00.478639290528629295"
  data-label="Send Feedback Button"
  track-type="feedback"
  track-name="sendFeedbackLink"
  track-metadata-position="thumb-rating"
  class="nocontent"
  data-nosnippet
  
  
  
    project-icon="https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers/images/touchicon-180-new.png"
  
  
  
  >

  <button>
  
    Need to tell us more?
  
  </button>
</devsite-feedback>
    </template>
  
  
    <template class="devsite-content-data-template">
      [[["Easy to understand","easyToUnderstand","thumb-up"],["Solved my problem","solvedMyProblem","thumb-up"],["Other","otherUp","thumb-up"]],[["Missing the information I need","missingTheInformationINeed","thumb-down"],["Too complicated / too many steps","tooComplicatedTooManySteps","thumb-down"],["Out of date","outOfDate","thumb-down"],["Samples / code issue","samplesCodeIssue","thumb-down"],["Other","otherDown","thumb-down"]],["Last updated 2026-05-28 UTC."],[],[]]
    </template>
  
</div>
            
          </devsite-content>
        </main>
        <devsite-footer-promos class="devsite-footer">
          
            
          
        </devsite-footer-promos>
        <devsite-footer-linkboxes class="devsite-footer">
          
            
<nav class="devsite-footer-linkboxes nocontent"
     aria-label="Footer links"
     data-nosnippet>
  
  <ul class="devsite-footer-linkboxes-list">
    
    <li class="devsite-footer-linkbox ">
    <h3 class="devsite-footer-linkbox-heading no-link">Connect</h3>
      <ul class="devsite-footer-linkbox-list">
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="//googledevelopers.blogspot.com"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 1)"
            >
            
          
            Blog
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="https://goo.gle/3FReQXN"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 2)"
            >
            
          
            Bluesky
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="https://www.instagram.com/googlefordevs/"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 3)"
            >
            
          
            Instagram
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="https://www.linkedin.com/showcase/googledevelopers/"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 4)"
            >
            
          
            LinkedIn
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="//twitter.com/googledevs"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 5)"
            >
            
          
            X (Twitter)
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="//www.youtube.com/user/GoogleDevelopers"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 6)"
            >
            
              
              
            
          
            YouTube
          
          </a>
          
          
        </li>
        
      </ul>
    </li>
    
    <li class="devsite-footer-linkbox ">
    <h3 class="devsite-footer-linkbox-heading no-link">Programs</h3>
      <ul class="devsite-footer-linkbox-list">
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="/program"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 1)"
            >
            
          
            Google Developer Program
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="/community"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 2)"
            >
            
          
            Google Developer Groups
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="/community/experts"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 3)"
            >
            
          
            Google Developer Experts
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="/community/accelerators"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 4)"
            >
            
          
            Accelerators
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="/community/nvidia"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 5)"
            >
            
              
              
            
          
            Google Cloud & NVIDIA
          
          </a>
          
          
        </li>
        
      </ul>
    </li>
    
    <li class="devsite-footer-linkbox ">
    <h3 class="devsite-footer-linkbox-heading no-link">Developer consoles</h3>
      <ul class="devsite-footer-linkbox-list">
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="//console.developers.google.com"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 1)"
            >
            
          
            Google API Console
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="//console.cloud.google.com"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 2)"
            >
            
          
            Google Cloud Platform Console
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="//play.google.com/apps/publish"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 3)"
            >
            
          
            Google Play Console
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="//console.firebase.google.com"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 4)"
            >
            
          
            Firebase Console
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="//console.actions.google.com"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 5)"
            >
            
          
            Actions on Google Console
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="//cast.google.com/publish"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 6)"
            >
            
          
            Cast SDK Developer Console
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="//chrome.google.com/webstore/developer/dashboard"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 7)"
            >
            
          
            Chrome Web Store Dashboard
          
          </a>
          
          
        </li>
        
        <li class="devsite-footer-linkbox-item">
          
          <a href="//console.home.google.com"
             class="devsite-footer-linkbox-link gc-analytics-event"
             data-category="Site-Wide Custom Events"
            
             data-label="Footer Link (index 8)"
            >
            
              
              
            
          
            Google Home Developer Console
          
          </a>
          
          
        </li>
        
      </ul>
    </li>
    
  </ul>
  
</nav>
          
        </devsite-footer-linkboxes>
        <devsite-footer-utility class="devsite-footer">
          
            

<div class="devsite-footer-utility nocontent" data-nosnippet>
  
  
  <nav class="devsite-footer-sites" aria-label="Other Google Developers websites">
    <a href="https://developers.google.com/"
       class="devsite-footer-sites-logo-link gc-analytics-event"
       data-category="Site-Wide Custom Events"
       data-label="Footer Google Developers Link">
      <picture>
        
        <img class="devsite-footer-sites-logo"
             src="https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers/images/lockup-google-for-developers.svg"
             loading="lazy"
             alt="Google Developers">
      </picture>
    </a>
    <ul class="devsite-footer-sites-list">
      
      <li class="devsite-footer-sites-item">
        <a href="//developer.android.com"
           class="devsite-footer-sites-link
                  gc-analytics-event"
           data-category="Site-Wide Custom Events"
         
           data-label="Footer Android Link"
         
         >
          Android
        </a>
      </li>
      
      <li class="devsite-footer-sites-item">
        <a href="//developer.chrome.com/home"
           class="devsite-footer-sites-link
                  gc-analytics-event"
           data-category="Site-Wide Custom Events"
         
           data-label="Footer Chrome Link"
         
         >
          Chrome
        </a>
      </li>
      
      <li class="devsite-footer-sites-item">
        <a href="//firebase.google.com"
           class="devsite-footer-sites-link
                  gc-analytics-event"
           data-category="Site-Wide Custom Events"
         
           data-label="Footer Firebase Link"
         
         >
          Firebase
        </a>
      </li>
      
      <li class="devsite-footer-sites-item">
        <a href="//cloud.google.com"
           class="devsite-footer-sites-link
                  gc-analytics-event"
           data-category="Site-Wide Custom Events"
         
           data-label="Footer Google Cloud Platform Link"
         
         >
          Google Cloud Platform
        </a>
      </li>
      
      <li class="devsite-footer-sites-item">
        <a href="//ai.google.dev/"
           class="devsite-footer-sites-link
                  gc-analytics-event"
           data-category="Site-Wide Custom Events"
         
           data-label="Footer Google AI Link"
         
         >
          Google AI
        </a>
      </li>
      
      <li class="devsite-footer-sites-item">
        <a href="/products"
           class="devsite-footer-sites-link
                  gc-analytics-event"
           data-category="Site-Wide Custom Events"
         
           data-label="Footer All products Link"
         
         >
          All products
        </a>
      </li>
      
    </ul>
  </nav>
  

  
  <nav class="devsite-footer-utility-links" aria-label="Utility links">
    
    <ul class="devsite-footer-utility-list">
      
      <li class="devsite-footer-utility-item
                 ">
        
        
        <a class="devsite-footer-utility-link gc-analytics-event"
           href="/terms/site-terms"
           data-category="Site-Wide Custom Events"
           data-label="Footer Terms link"
         >
          Terms
        </a>
        
      </li>
      
      <li class="devsite-footer-utility-item
                 ">
        
        
        <a class="devsite-footer-utility-link gc-analytics-event"
           href="//policies.google.com/privacy"
           data-category="Site-Wide Custom Events"
           data-label="Footer Privacy link"
         >
          Privacy
        </a>
        
      </li>
      
      <li class="devsite-footer-utility-item
                 glue-cookie-notification-bar-control">
        
        
        <a class="devsite-footer-utility-link gc-analytics-event"
           href="#"
           data-category="Site-Wide Custom Events"
           data-label="Footer Manage cookies link"
         
           aria-hidden="true"
         >
          Manage cookies
        </a>
        
      </li>
      
    </ul>
    
    
<devsite-language-selector>
  <ul role="presentation">
    
    
    <li role="presentation">
      <a role="menuitem" lang="en"
        >English</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="de"
        >Deutsch</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="es"
        >Español</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="fr"
        >Français</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="id"
        >Indonesia</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="pt_br"
        >Português – Brasil</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="ru"
        >Русский</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="zh_cn"
        >中文 – 简体</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="ja"
        >日本語</a>
    </li>
    
    <li role="presentation">
      <a role="menuitem" lang="ko"
        >한국어</a>
    </li>
    
  </ul>
</devsite-language-selector>

  </nav>
</div>
          
        </devsite-footer-utility>
        <devsite-panel>
          
        </devsite-panel>
        
          <devsite-concierge
  
  
    data-info-panel
  
  
    data-ai-panel
  
  
  
  
    data-api-explorer-panel
  >
</devsite-concierge>
        
      </section>
      </section>
    <devsite-sitemask></devsite-sitemask>
    <devsite-snackbar></devsite-snackbar>
    <devsite-tooltip ></devsite-tooltip>
    <devsite-heading-link></devsite-heading-link>
    <devsite-analytics>
      
        <script type="application/json" analytics>[{&#34;dimensions&#34;: {&#34;dimension5&#34;: &#34;en&#34;, &#34;dimension3&#34;: false, &#34;dimension4&#34;: &#34;Google AI Edge&#34;, &#34;dimension6&#34;: &#34;en&#34;, &#34;dimension11&#34;: false, &#34;dimension1&#34;: &#34;Signed out&#34;}, &#34;gaid&#34;: &#34;UA-24532603-1&#34;, &#34;metrics&#34;: {&#34;ratings_value&#34;: &#34;metric1&#34;, &#34;ratings_count&#34;: &#34;metric2&#34;}, &#34;purpose&#34;: 1}]</script>
<script type="application/json" tag-management>{&#34;at&#34;: &#34;True&#34;, &#34;ga4&#34;: [{&#34;id&#34;: &#34;G-272J68FCRF&#34;, &#34;purpose&#34;: 1}], &#34;ga4p&#34;: [{&#34;id&#34;: &#34;G-272J68FCRF&#34;, &#34;purpose&#34;: 1}], &#34;gtm&#34;: [{&#34;id&#34;: &#34;GTM-T98GCPGN&#34;, &#34;purpose&#34;: 1}], &#34;parameters&#34;: {&#34;internalUser&#34;: &#34;False&#34;, &#34;language&#34;: {&#34;machineTranslated&#34;: &#34;False&#34;, &#34;requested&#34;: &#34;en&#34;, &#34;served&#34;: &#34;en&#34;}, &#34;pageType&#34;: &#34;landing&#34;, &#34;projectName&#34;: &#34;Google AI Edge&#34;, &#34;signedIn&#34;: &#34;False&#34;, &#34;tenant&#34;: &#34;developers&#34;, &#34;recommendations&#34;: {&#34;sourcePage&#34;: &#34;&#34;, &#34;sourceType&#34;: 0, &#34;sourceRank&#34;: 0, &#34;sourceIdenticalDescriptions&#34;: 0, &#34;sourceTitleWords&#34;: 0, &#34;sourceDescriptionWords&#34;: 0, &#34;experiment&#34;: &#34;&#34;}, &#34;experiment&#34;: {&#34;ids&#34;: &#34;&#34;}}}</script>
      
    </devsite-analytics>
    
      <devsite-badger></devsite-badger>
    
    
    
    
<script nonce="KCnaBOOM0h9HZ01PFiXZNvld+ApEZ5">
  
  (function(d,e,v,s,i,t,E){d['GoogleDevelopersObject']=i;
    t=e.createElement(v);t.async=1;t.src=s;E=e.getElementsByTagName(v)[0];
    E.parentNode.insertBefore(t,E);})(window, document, 'script',
    'https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers/js/app_loader.js', '[1,"en",null,"/js/devsite_app_module.js","https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd","https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers","https://developers-dot-devsite-v2-prod.appspot.com",1,null,["/_pwa/developers/manifest.json","https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/images/video-placeholder.svg","https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers/images/favicon-new.png","https://www.gstatic.com/devrel-devsite/prod/v3be1e30159846e100d05529400567b663b9f8b605137438a2f417848d68359dd/developers/images/lockup-new.svg","https://fonts.googleapis.com/css?family=Google+Sans:400,500|Roboto:400,400italic,500,500italic,700,700italic|Roboto+Mono:400,500,700&display=swap"],1,null,[1,6,8,12,14,17,21,25,50,52,63,70,75,76,80,87,91,92,93,97,98,100,101,102,103,104,105,107,108,109,110,112,113,116,117,118,120,122,124,125,126,127,129,130,131,132,133,134,135,136,138,140,141,147,148,149,151,152,156,157,158,159,161,163,164,168,169,170,179,180,182,183,186,191,193,196],"AIzaSyAP-jjEJBzmIyKR4F-3XITp8yM9T1gEEI8","AIzaSyB6xiKGDR5O3Ak2okS4rLkauxGUG7XP0hg","developers.google.com","AIzaSyAQk0fBONSGUqCNznf6Krs82Ap1-NV6J4o","AIzaSyCCxcqdrZ_7QMeLCRY20bh_SXdAYqy70KY",null,null,null,["Profiles__enable_completecodelab_endpoint","MiscFeatureFlags__enable_explain_this_code","DevPro__enable_payments_first_batch","EngEduTelemetry__enable_engedu_telemetry","Profiles__enable_awarding_url","Cloud__cache_serialized_dynamic_content","DevPro__enable_vertex_credit_card","Analytics__enable_devpro_interaction_logging","Profiles__enable_public_developer_profiles","DevPro__enable_devpro_offers","Cloud__enable_legacy_calculator_redirect","Concierge__enable_actions_menu","CloudShell__cloud_code_overflow_menu","BookNav__enable_tenant_cache_key","MiscFeatureFlags__gdp_dashboard_reskin_enabled","Cloud__enable_cloudx_experiment_ids","Profiles__enable_recognition_badges","MiscFeatureFlags__enable_framebox_badge_methods","MiscFeatureFlags__enable_explicit_template_dependencies","Cloud__fast_free_trial","DevPro__enable_enterprise","Search__enable_page_map","MiscFeatureFlags__developers_footer_image","Profiles__enable_dashboard_curated_recommendations","Profiles__enable_callout_notifications","Profiles__enable_complete_playlist_endpoint","Concierge__enable_key_takeaways_new_ui","SignIn__enable_l1_signup_flow","AIStudioInteractionsToggle__interactions_are_default","Cloud__enable_llm_concierge_chat","DevPro__enable_developer_subscriptions","TpcFeatures__enable_unmirrored_page_left_nav","Profiles__enable_purchase_prompts","Search__enable_dynamic_content_confidential_banner","MiscFeatureFlags__enable_variable_operator","Profiles__enable_profile_collections","MiscFeatureFlags__remove_cross_domain_tracking_params","Concierge__enable_remove_info_panel_tags","Profiles__require_profile_eligibility_for_signin","DevPro__enable_google_one_card","Significatio__enable_by_tenant","DevPro__enable_firebase_workspaces_card","Search__enable_ai_search_summaries_restricted","DevPro__enable_cloud_innovators_plus","MiscFeatureFlags__enable_firebase_utm","Profiles__enable_join_program_group_endpoint","Concierge__enable_concierge_restricted","Search__enable_ai_eligibility_checks","DevPro__enable_free_benefits","DevPro__enable_credits_banner","MiscFeatureFlags__enable_variable_operator_index_yaml","DevPro__enable_google_payments_buyflow","Profiles__enable_completequiz_endpoint","Profiles__enable_auto_apply_credits","DevPro__enable_embed_profile_creation","Profiles__enable_developer_profiles_callout","MiscFeatureFlags__enable_project_variables","CloudShell__cloud_shell_button","Search__enable_ai_search_summaries_for_all","Experiments__reqs_query_experiments","Cloud__enable_free_trial_server_call","Analytics__enable_clearcut_logging","DevPro__enable_g1_ineligible_redirect","DevPro__remove_eu_tax_intake_form","Concierge__enable_concierge","MiscFeatureFlags__developers_footer_dark_image","Profiles__enable_user_type","DevPro__enable_code_assist","Cloud__enable_cloud_shell","Cloud__enable_cloud_dlp_service","DevPro__enable_nvidia_credits_card","Profiles__enable_playlist_community_acl","Concierge__enable_pushui","Profiles__enable_targeted_hero","Search__enable_suggestions_from_borg","Profiles__enable_release_notes_notifications","Cloud__enable_cloud_shell_fte_user_flow","Search__enable_ai_search_summaries","Concierge__enable_devsite_llm_tools","MiscFeatureFlags__enable_appearance_cookies","DevPro__enable_devsite_captcha","Profiles__enable_developer_profile_benefits_ui_redesign","DevPro__enable_g1_integration","Profiles__enable_developer_profile_pages_as_content","TpcFeatures__proxy_prod_host","MiscFeatureFlags__enable_view_transitions","MiscFeatureFlags__fix_lower_breadcrumbs","Profiles__enable_page_saving","Concierge__enable_key_takeaways","Profiles__enable_stripe_subscription_management"],null,null,"AIzaSyBLEMok-5suZ67qRPzx0qUtbnLmyT_kCVE","https://developerscontentserving-pa.clients6.google.com","AIzaSyCM4QpTRSqP5qI4Dvjt4OAScIN8sOUlO-k","https://developerscontentsearch-pa.clients6.google.com",1,4,null,"https://developerprofiles-pa.clients6.google.com",[1,"developers","Google for Developers","developers.google.com",null,"developers-dot-devsite-v2-prod.appspot.com",null,null,[1,1,[1],null,null,null,null,null,null,null,null,[1],null,null,null,null,null,null,[1],[1,null,null,[1,20],"/recommendations/information"],null,null,null,[1,1,1],[1,1,null,1,1,null,null,["/meridian"]],[1,null,null,null,null,null,null,["/admob","/ad-manager/mobile-ads-sdk","/ml-kit"]],null,[null,["/meridian","/youtube/devices/"]],1,null,[1]],null,[null,null,null,null,null,null,"/images/lockup-new.svg","/images/touchicon-180-new.png",null,null,null,null,1,null,null,null,null,null,null,null,null,1,null,null,null,"/images/lockup-dark-theme-new.svg",[]],[],null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,[6,1,14,15,20,22,23,29,32,36],null,[[null,null,null,[3,7,10,2,39,17,4,32,24,11,12,13,34,15,25],null,null,[1,[["docType","Choose a content type",[["Tutorial",null,null,null,null,null,null,null,null,"Tutorial"],["Guide",null,null,null,null,null,null,null,null,"Guide"],["Sample",null,null,null,null,null,null,null,null,"Sample"]]],["product","Choose a product",[["Android",null,null,null,null,null,null,null,null,"Android"],["ARCore",null,null,null,null,null,null,null,null,"ARCore"],["ChromeOS",null,null,null,null,null,null,null,null,"ChromeOS"],["Firebase",null,null,null,null,null,null,null,null,"Firebase"],["Flutter",null,null,null,null,null,null,null,null,"Flutter"],["Assistant",null,null,null,null,null,null,null,null,"Google Assistant"],["GoogleCloud",null,null,null,null,null,null,null,null,"Google Cloud"],["GoogleMapsPlatform",null,null,null,null,null,null,null,null,"Google Maps Platform"],["GooglePay",null,null,null,null,null,null,null,null,"Google Pay & Google Wallet"],["GooglePlay",null,null,null,null,null,null,null,null,"Google Play"],["Tensorflow",null,null,null,null,null,null,null,null,"TensorFlow"]]],["category","Choose a topic",[["AiAndMachineLearning",null,null,null,null,null,null,null,null,"AI and Machine Learning"],["Data",null,null,null,null,null,null,null,null,"Data"],["Enterprise",null,null,null,null,null,null,null,null,"Enterprise"],["Gaming",null,null,null,null,null,null,null,null,"Gaming"],["Mobile",null,null,null,null,null,null,null,null,"Mobile"],["Web",null,null,null,null,null,null,null,null,"Web"]]]]]],[1,1],null,1],[[["UA-24532603-1"],["UA-22084204-5"],null,null,["UA-24532603-5"],["GTM-T98GCPGN"],null,[["G-272J68FCRF"],null,null,[["G-272J68FCRF",2]]],[["UA-24532603-1",2]],null,[["UA-24532603-5",2]],[["GTM-T98GCPGN",2]],1],[[15,12],[4,3],[5,4],[3,2],[14,11],[12,9],[1,1],[16,13],[11,8],[6,5],[13,10]],[[2,2],[1,1]]],null,4,null,null,null,null,null,null,null,null,null,null,null,null,null,"developers.devsite.google",null,null,null,null,null,[]],null,"pk_live_5170syrHvgGVmSx9sBrnWtA5luvk9BwnVcvIi7HizpwauFG96WedXsuXh790rtij9AmGllqPtMLfhe2RSwD6Pn38V00uBCydV4m",1,1,"https://developerscontentinsights-pa.clients6.google.com","AIzaSyCg-ZUslalsEbXMfIo9ZP8qufZgo3LSBDU","AIzaSyDxT0vkxnY_KeINtA4LSePJO-4MAZPMRsE","https://developers.clients6.google.com",["https://codeassist.google.com","https://code-assist-free-tier.corp.google.com"],null,"AIzaSyBQom12tzI-rybN7Sf-KfeL4nwm-Rf7PmI\n",null,null,"https://developers.googleapis.com"]')
  
</script>

    <devsite-a11y-announce></devsite-a11y-announce>
  </body>
</html>