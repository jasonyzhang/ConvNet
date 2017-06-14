require('es6-promise/auto');  // polyfill Promise on IE

var PageConfig = require('@jupyterlab/coreutils').PageConfig;
__webpack_public_path__ = PageConfig.getOption('publicUrl');

// This needs to come after __webpack_public_path__ is set.
require('font-awesome/css/font-awesome.min.css');
// Load the core theming before any other package.
require('@jupyterlab/theming/style/index.css');

var app = require('@jupyterlab/application').JupyterLab;

function main() {
    var version = PageConfig.getOption('appVersion') || 'unknown';
    var name = PageConfig.getOption('appName') || 'JupyterLab';
    var namespace = PageConfig.getOption('appNamespace') || 'jupyterlab';
    var devMode = PageConfig.getOption('devMode') || 'false';
    var settingsDir = PageConfig.getOption('settingsDir') || '';
    var assetsDir = PageConfig.getOption('assetsDir') || '';

    if (version[0] === 'v') {
        version = version.slice(1);
    }

    lab = new app({
        namespace: namespace,
        name: name,
        version: version,
        devMode: devMode.toLowerCase() === 'true',
        settingsDir: settingsDir,
        assetsDir: assetsDir
    });
    try {
        lab.registerPluginModule(require('@jupyterlab/application-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/apputils-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/chatbox-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/codemirror-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/completer-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/console-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/csvviewer-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/docmanager-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/docregistry-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/fileeditor-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/faq-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/filebrowser-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/help-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/imageviewer-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/inspector-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/landing-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/launcher-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/markdownviewer-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/notebook-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/rendermime-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/running-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/services-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/shortcuts-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/tabmanager-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/terminal-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/theme-light-extension'));
    } catch (e) {
        console.error(e);
    }
    try {
        lab.registerPluginModule(require('@jupyterlab/tooltip-extension'));
    } catch (e) {
        console.error(e);
    }
    var ignorePlugins = [];
    try {
        var option = PageConfig.getOption('ignorePlugins');
        ignorePlugins = JSON.parse(option);
    } catch (e) {
        console.error("Invalid ignorePlugins config:", option);
    }
    lab.start({ "ignorePlugins": ignorePlugins });
}

window.onload = main;
