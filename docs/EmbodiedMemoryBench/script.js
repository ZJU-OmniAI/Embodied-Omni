'use strict';
const families = {
  po: { skill:'Fine-grained visual memory',title:'Remember the detail before it matters.',description:'The future task is unknown during observation. Can the agent retain an object’s location amid a busy scene and unrelated activity?',frames:[{image:27,label:'Earlier observation',title:'Explore the room',text:'The agent encounters unrelated objects and scenes.'},{image:18,label:'Memory cue',title:'Notice the laptop',text:'A laptop is visible on the coffee table among other objects.'},{image:28,label:'Distractor activity',title:'Move on to other tasks',text:'Unrelated observations separate the cue from the later task.'}],task:'“Pick up the laptop.”',answer:'Return to the coffee table in the living room. The earlier observation grounds the laptop’s location.'},
  dt: { skill:'Dynamic world-state tracking',title:'Keep the latest state, not the first.',description:'An object changes location during the history. The agent must update its memory and act on the most recent evidence.',frames:[{image:0,label:'Earlier state',title:'Book on the bed',text:'An early observation places the book on the bed.'},{image:1,label:'State update',title:'Book in the box',text:'The book moves. Its old location is no longer valid.'},{image:2,label:'Latest state',title:'Book on the desk',text:'The latest observation supersedes both earlier locations.'}],task:'“Find the book and bring it to the table.”',answer:'Navigate to the desk to find the book. Returning to the bed would use an outdated world state.'},
  if: { skill:'Interaction-derived world state',title:'A failed action is useful evidence.',description:'Some states are invisible until an interaction reveals them. The agent must retain failure feedback and choose a viable alternative later.',frames:[{image:11,label:'Interaction feedback',title:'Top drawer is locked',text:'An attempted opening fails and reveals a hidden constraint.'},{image:27,label:'Distractor activity',title:'Continue exploring',text:'Other activity intervenes before the drawer is needed again.'},{image:12,label:'Viable alternative',title:'Bottom drawer opens',text:'The history provides evidence of an accessible alternative.'}],task:'“Put the keychain into a drawer.”',answer:'Open the usable bottom drawer. Remembering the earlier feedback avoids another attempt on the locked top drawer.'},
  eg: { skill:'Experience generalization',title:'Turn past corrections into a new action.',description:'Several corrected experiences reveal a shared placement rule. The target task introduces a different object from the same category.',frames:[{image:20,label:'Correction 1',title:'Knife → drawer',text:'A correction teaches where a knife should be stored.'},{image:24,label:'Correction 2',title:'Spoon → drawer',text:'A second case reinforces the same household regularity.'},{image:36,label:'New task object',title:'Now encounter a fork',text:'The fork is new; the learned regularity must transfer.'}],task:'“Please store this fork in a proper place.”',answer:'Place the fork in the kitchen drawer. Apply the shared flatware-storage rule learned from earlier corrections.'}
};
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
function selectFamily(key) {
  const family=families[key];
  $$('.family-tab').forEach(tab=>{const active=tab.dataset.family===key;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;});
  $('#family-panel').setAttribute('aria-labelledby',`tab-${key}`);
  $('#family-skill').textContent=family.skill.toUpperCase();
  $('#family-title').textContent=family.title;
  $('#family-description').textContent=family.description;
  $('#trace-grid').replaceChildren(...family.frames.map(frame=>{
    const card=document.createElement('article');card.className='trace-card';
    const visual=document.createElement('div');visual.className='trace-image';
    const img=document.createElement('img');img.src=`assets/scene-${String(frame.image).padStart(2,'0')}.png`;img.alt=frame.title;img.width=500;img.height=500;
    const label=document.createElement('span');label.className='trace-step';label.textContent=frame.label;
    visual.append(img,label);
    const title=document.createElement('h4');title.textContent=frame.title;
    const text=document.createElement('p');text.textContent=frame.text;
    card.append(visual,title,text);return card;
  }));
  $('#task-instruction').textContent=family.task;
  $('#answer-text').textContent=family.answer;
  $('#answer-content').hidden=true;
  $('#reveal-answer').hidden=false;
  $('#reveal-answer').setAttribute('aria-expanded','false');
}
$$('.family-tab').forEach((tab,index)=>{
  tab.addEventListener('click',()=>selectFamily(tab.dataset.family));
  tab.addEventListener('keydown',event=>{
    const tabs=$$('.family-tab');let next;
    if(event.key==='ArrowRight')next=(index+1)%tabs.length;
    if(event.key==='ArrowLeft')next=(index+tabs.length-1)%tabs.length;
    if(event.key==='Home')next=0;
    if(event.key==='End')next=tabs.length-1;
    if(next!==undefined){event.preventDefault();selectFamily(tabs[next].dataset.family);tabs[next].focus();}
  });
});
$$('[data-select-family]').forEach(link=>link.addEventListener('click',()=>selectFamily(link.dataset.selectFamily)));
$('#reveal-answer').addEventListener('click',()=>{
  $('#answer-content').hidden=false;
  $('#reveal-answer').setAttribute('aria-expanded','true');
  $('#reveal-answer').hidden=true;
  $('#answer-content').tabIndex=-1;$('#answer-content').focus({preventScroll:true});
});
selectFamily('po');

let metric='sr',sortIndex=0,ascending=false;
const familyNames=['Passive Observation','Dynamic Tracking','Interaction Failure','Experience Generalization'];
const isSubset=row=>row.name==='Robotics-ER 1.5*';
const isBaseline=row=>row.group==='open'||row.group==='proprietary'||row.group==='embodied'||row.name==='Full context';
function settingInfo(row){
  if(row.name==='EMem-8B')return {label:'EMem + SFT',tone:'trained',detail:'Qwen3-VL-8B · trained policy'};
  if(row.ours)return {label:'EMem',tone:'emem',detail:row.group==='memory'?'GPT-5.4-mini backbone':'External memory'};
  if(isBaseline(row))return {label:'Full context',tone:'context',detail:row.group==='memory'?'GPT-5.4-mini backbone':isSubset(row)?'10% evaluation subset':'Full-set evaluation'};
  return {label:'Memory system',tone:'memory',detail:'GPT-5.4-mini backbone'};
}
function node(tag,className,text){const el=document.createElement(tag);if(className)el.className=className;if(text!==undefined)el.textContent=text;return el;}
function heatColor(value,key){
  const quality=Math.max(0,Math.min(100,key==='err'?100-value:value))/100;
  return `hsl(${10+quality*158} ${46+quality*10}% ${93-quality*23}%)`;
}
function modelCell(row,className){
  const cell=node('th',className);cell.scope='row';
  const title=node('div','model-title',row.name);if(row.ours)title.append(node('span','ours-badge','Ours'));
  cell.append(title,node('span','model-detail',settingInfo(row).detail));return cell;
}
function scoreChip(value,key,best,primary){
  const chip=node('span',`score-chip${best?' is-best':''}${primary?' is-primary':''}`);
  chip.style.setProperty('--score-color',heatColor(value,key));chip.style.setProperty('--score-width',`${value}%`);
  chip.append(node('span','',value.toFixed(1)));
  chip.title=`${value.toFixed(1)}% ${key.toUpperCase()} · ${key==='sr'?'higher':'lower'} is better${best?' · best in reported full-set results':''}`;
  if(best)chip.setAttribute('aria-label',chip.title);
  return chip;
}
function bestValue(rows,key,index){return (key==='sr'?Math.max:Math.min)(...rows.map(row=>row[key][index]));}
function winnersFor(rows,key,index){const best=bestValue(rows,key,index);return {value:best,rows:rows.filter(row=>row[key][index]===best)};}
function winnerLabel(rows){return rows.map(row=>row.name+(row.group==='emem'&&row.name!=='EMem-8B'?' + EMem':'')).join(' / ');}
function renderSummary(eligible){
  const overall=winnersFor(eligible,metric,0);
  const card=node('article','summary-overall');
  card.append(node('span','summary-label',`Best average ${metric.toUpperCase()} ${metric==='sr'?'↑':'↓'}`),node('strong','summary-value',overall.value.toFixed(1)),node('p','summary-model',winnerLabel(overall.rows)),node('span','summary-scope','Across reported full-set settings'));
  $('#results-summary').replaceChildren(card);
}
function renderResults(){
  const selected=window.EMEM_RESULTS;
  const eligible=selected.filter(row=>!isSubset(row));
  const rows=[...selected].sort((a,b)=>(a[metric][sortIndex]-b[metric][sortIndex])*(ascending?1:-1));
  const best=Array.from({length:5},(_,index)=>bestValue(eligible,metric,index));
  $('#table-context').textContent='All backbones and memory systems · sorted together by the selected metric.';
  $('#result-count').replaceChildren(node('strong','',String(rows.length)),node('span','','settings'));
  $('#rank-metric').textContent=metric.toUpperCase();
  renderSummary(eligible);
  $('#results-body').replaceChildren(...rows.map(row=>{
    const tr=node('tr',isSubset(row)?'subset-row':row.ours?'ours':'');tr.dataset.model=row.name;tr.dataset.group=row.group;
    const rank=node('td','rank-cell');
    if(isSubset(row)){const mark=node('span','subset-rank','—');mark.title='Subset result; excluded from full-set ranking';rank.append(mark);}
    else{const position=1+eligible.filter(other=>metric==='sr'?other.sr[0]>row.sr[0]:other.err[0]<row.err[0]).length;const mark=node('span',`rank-medal${position<=3?' place-'+position:''}`,String(position));mark.setAttribute('aria-label',`Rank ${position} by average ${metric.toUpperCase()}`);rank.append(mark);}
    const info=settingInfo(row);const setting=node('td','setting-cell');setting.append(node('span',`setting-badge ${info.tone}`,info.label));
    tr.append(rank,modelCell(row,'model-cell'),setting);
    row[metric].forEach((value,index)=>{const cell=node('td',index===0?'score-td average-td':'score-td');cell.append(scoreChip(value,metric,!isSubset(row)&&value===best[index],index===0));tr.append(cell);});return tr;
  }));
  $$('[data-sort]').forEach(button=>{const active=Number(button.dataset.sort)===sortIndex;button.parentElement.removeAttribute('aria-sort');if(active)button.parentElement.setAttribute('aria-sort',ascending?'ascending':'descending');button.querySelector('span').textContent=active?(ascending?'↑':'↓'):'↕';});
}
$('#metric-select').addEventListener('change',event=>{metric=event.target.value;ascending=metric==='err';renderResults();});
$$('[data-sort]').forEach(button=>button.addEventListener('click',()=>{const index=Number(button.dataset.sort);ascending=index===sortIndex?!ascending:metric==='err';sortIndex=index;renderResults();}));
renderResults();
$('#download-results').addEventListener('click',()=>{
  const fields=['Model','Setting','Average SR','Passive SR','Dynamic SR','Interaction SR','Experience SR','Average ERR','Passive ERR','Dynamic ERR','Interaction ERR','Experience ERR'];
  const csv=[fields,...window.EMEM_RESULTS.map(row=>[row.name,row.group,...row.sr,...row.err])].map(row=>row.map(value=>`"${String(value).replaceAll('"','""')}"`).join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8;'}));const link=document.createElement('a');link.href=url;link.download='emem-bench-paper-results.csv';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
});

const dialog=$('#figure-dialog');
$$('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{
  $('#dialog-image').src=button.dataset.zoom;$('#dialog-image').alt=button.dataset.caption;$('#dialog-caption').textContent=button.dataset.caption;dialog.showModal();document.body.style.overflow='hidden';
}));
$('#close-dialog').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target===dialog){const box=dialog.getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)dialog.close();}});
dialog.addEventListener('close',()=>{document.body.style.overflow='';});
let toastTimer;
function showToast(text){const toast=$('#toast');toast.textContent=text;toast.classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>toast.classList.remove('visible'),3000);}
$('#copy-command').addEventListener('click',async()=>{
  const command=$('#install-command').textContent;
  try{
    if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(command);
    else{const textarea=document.createElement('textarea');textarea.value=command;textarea.style.position='fixed';textarea.style.opacity='0';document.body.append(textarea);textarea.select();const copied=document.execCommand('copy');textarea.remove();if(!copied)throw new Error('Copy unavailable');}
    showToast('Commands copied to clipboard');
  }catch{const selection=window.getSelection();const range=document.createRange();range.selectNodeContents($('#install-command'));selection.removeAllRanges();selection.addRange(range);showToast('Commands selected — press Ctrl+C or ⌘C to copy');}
});
const menu=$('.menu-toggle'),navigation=$('#navigation');
function closeMenu(){menu.setAttribute('aria-expanded','false');menu.setAttribute('aria-label','Open navigation');navigation.classList.remove('open');}
menu.addEventListener('click',()=>{const expanded=menu.getAttribute('aria-expanded')!=='true';menu.setAttribute('aria-expanded',String(expanded));menu.setAttribute('aria-label',expanded?'Close navigation':'Open navigation');navigation.classList.toggle('open',expanded);});
$$('#navigation a').forEach(link=>link.addEventListener('click',closeMenu));
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&navigation.classList.contains('open')){closeMenu();menu.focus();}});
const navLinks=$$('#navigation a');
const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting)navLinks.forEach(link=>{const active=link.hash===`#${entry.target.id}`;link.classList.toggle('active',active);if(active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});});},{rootMargin:'-15% 0px -65% 0px',threshold:0});
$$('main section[id]').forEach(section=>observer.observe(section));
